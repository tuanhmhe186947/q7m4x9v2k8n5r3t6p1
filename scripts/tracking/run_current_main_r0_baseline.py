"""Run the exact-current-main replay-only full-13 R0 tracking baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "tracking"))

from generate_current_main_baseline_caches import (  # noqa: E402
    CREATION_AUTHORITY,
    STARTING_MAIN_SHA,
    VideoAuthority,
    cache_path,
    canonical_hash,
    detector_configuration,
    effective_config_payload,
    frame_indices,
    load_population,
    sha256_file,
)

PROFILE = "realtime_fast"
IOU_THRESHOLD = 0.5
GAP_TOLERANCE_FRAMES = 15


class R0BaselineError(RuntimeError):
    """Fail-closed R0 cache, replay, or evaluation error."""


def write_json(path: Path, payload: Any) -> None:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise R0BaselineError(f"refusing empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    ).strip()


def require_exact_tracking_authority() -> tuple[str, str]:
    head = git_output("rev-parse", "HEAD")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_MAIN_SHA, head],
        cwd=REPO,
        check=False,
    )
    if result.returncode:
        raise R0BaselineError("starting-main authority is not an ancestor")
    starting_tree = git_output(
        "rev-parse",
        f"{STARTING_MAIN_SHA}:src/pig_behavior/tracking",
    )
    current_tree = git_output("rev-parse", "HEAD:src/pig_behavior/tracking")
    if current_tree != starting_tree:
        raise R0BaselineError("tracking subtree differs from starting main")
    if git_output("status", "--short"):
        raise R0BaselineError("R0 producer worktree must be clean")
    return head, current_tree


def cache_identity(video: VideoAuthority, decision: dict[str, Any]) -> Any:
    from pig_behavior.tracking.detector_cache import DetectorCacheIdentity

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=decision["detector_weight_sha256"],
        detector_semantic_config_sha256=decision[
            "detector_semantic_config_sha256"
        ],
        producer_code_sha=decision["producer_code_sha"],
        creation_authority=CREATION_AUTHORITY,
    )


def load_caches(
    videos: list[VideoAuthority],
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pig_behavior.tracking.detector_cache import (
        DETECTOR_CACHE_SCHEMA_VERSION,
        DetectorEvidenceCache,
        ReplayDetector,
    )

    decision_path = (
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_GENERATION_DECISION.json"
    )
    replay_path = (
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_REPLAY_VALIDATION.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    replay_authority = json.loads(replay_path.read_text(encoding="utf-8"))
    if decision["decision"] != "PASS_EXACT_CURRENT_MAIN_BASELINE_CACHES_READY":
        raise R0BaselineError("cache-generation decision did not pass")
    if replay_authority["result"] != "PASS":
        raise R0BaselineError("cache replay authority did not pass")
    if decision["cache_schema"] != DETECTOR_CACHE_SCHEMA_VERSION:
        raise R0BaselineError("cache schema differs from live main")
    caches: dict[str, Any] = {}
    cache_hashes: dict[str, str] = {}
    loaded_frames = 0
    for video in videos:
        path = cache_path(cache_root, video.video_key)
        expected_sha = decision["cache_artifacts"][video.video_key]["sha256"]
        if sha256_file(path) != expected_sha:
            raise R0BaselineError(f"cache hash mismatch: {video.video_key}")
        cache = DetectorEvidenceCache.load(
            path,
            expected_identity=cache_identity(video, decision),
        )
        expected_frames = frame_indices(video, 2)
        if tuple(cache.frames) != expected_frames:
            raise R0BaselineError(
                f"cache coverage mismatch: {video.video_key}"
            )
        replay = ReplayDetector(cache)
        for frame_index in expected_frames:
            replay.set_frame_context(
                frame_index,
                cache.frames[frame_index]["original_frame_dimensions"],
            )
            replay.predict()
        if replay.invocations != 900:
            raise R0BaselineError("record-load-replay count mismatch")
        caches[video.video_key] = cache
        cache_hashes[video.video_key] = expected_sha
        loaded_frames += replay.invocations
    if loaded_frames != 11700:
        raise R0BaselineError("full-13 cache population is not 11,700")
    return caches, {
        "result": "PASS",
        "cache_frames_loaded": loaded_frames,
        "detector_inference_calls": 0,
        "cache_schema": DETECTOR_CACHE_SCHEMA_VERSION,
        "cache_hashes": cache_hashes,
        "generation_decision_sha256": sha256_file(decision_path),
        "replay_authority_sha256": sha256_file(replay_path),
    }


def build_cfg(source_repo: Path, video: VideoAuthority, output_dir: Path) -> Any:
    from pig_behavior.tracking.config import TrackingConfig, validate_config
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    cfg = TrackingConfig(
        mode="realtime",
        video_path=video.video_path,
        weights_path=(
            source_repo / "models" / "detector" / "pig_detector_yolov8.pt"
        ),
        mask_path=source_repo / "data" / "annotations" / "scene" / "mask.png",
        output_dir=output_dir,
        device="cpu",
        half=False,
        write_output_video=False,
        start_frame=0,
        max_frames=video.frame_count,
        **EVAL_CONFIGS[PROFILE],
    )
    cfg.association_debug = False
    validate_config(cfg)
    return cfg


def evaluate_video(video: VideoAuthority, prediction_xml: Path) -> Any:
    from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml
    from pig_behavior.evaluation.tracking.evaluator import evaluate_tracking
    from pig_behavior.evaluation.tracking.metrics import (
        attach_remapped_metrics,
        remap_prediction_ids,
    )

    ground_truth = parse_cvat_video_xml(video.gt_path, include_hidden=True)
    prediction = parse_cvat_video_xml(prediction_xml, include_hidden=True)
    metrics = evaluate_tracking(
        ground_truth,
        prediction,
        iou_threshold=IOU_THRESHOLD,
        video_stem=video.video_key,
        gap_tolerance_frames=GAP_TOLERANCE_FRAMES,
    )
    remapped_prediction, _mapping, mapped_matches, coverage = (
        remap_prediction_ids(
            ground_truth,
            prediction,
            iou_threshold=IOU_THRESHOLD,
        )
    )
    remapped = evaluate_tracking(
        ground_truth,
        remapped_prediction,
        iou_threshold=IOU_THRESHOLD,
        video_stem=video.video_key,
        gap_tolerance_frames=GAP_TOLERANCE_FRAMES,
    )
    attach_remapped_metrics(
        metrics,
        remapped,
        mapped_matches=mapped_matches,
        coverage=coverage,
    )
    metrics.gt_xml = str(video.gt_path)
    metrics.pred_xml = str(prediction_xml)
    metrics.video_path = str(video.video_path)
    return metrics


def run_predictions(
    source_repo: Path,
    videos: list[VideoAuthority],
    caches: dict[str, Any],
    output_root: Path,
) -> tuple[list[Any], int]:
    from pig_behavior.tracking.detector_cache import ReplayDetector
    from pig_behavior.tracking.runner import run_tracking

    metrics = []
    replay_calls = 0
    for video in videos:
        print(f"R0_BEGIN {video.video_key}", flush=True)
        output_dir = output_root / "predictions" / video.video_key
        detector = ReplayDetector(caches[video.video_key])
        summary = run_tracking(
            build_cfg(source_repo, video, output_dir),
            model=detector,
        )
        if detector.invocations != 900:
            raise R0BaselineError(
                f"tracking replay mismatch: {video.video_key}"
            )
        metrics.append(
            evaluate_video(video, Path(summary.cvat_video_xml))
        )
        replay_calls += detector.invocations
        print(
            f"R0_END {video.video_key} "
            f"idsw={metrics[-1].remapped_idsw}",
            flush=True,
        )
    if replay_calls != 11700:
        raise R0BaselineError("R0 did not replay all 11,700 detector frames")
    return metrics, replay_calls


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "ARTIFACT_SHA256.json"
    ]


def execute(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    output_root: Path,
) -> None:
    from pig_behavior.evaluation.tracking.metrics import aggregate_metrics

    if output_root.exists():
        raise R0BaselineError(f"refusing existing output root: {output_root}")
    producer_sha, tracking_tree = require_exact_tracking_authority()
    videos, lineage_file_sha = load_population(source_repo, lineage_manifest)
    cfg, detector_payload = detector_configuration(source_repo)
    config_payload = effective_config_payload()
    caches, cache_report = load_caches(videos, cache_root)
    output_root.mkdir(parents=True)
    metrics, replay_calls = run_predictions(
        source_repo,
        videos,
        caches,
        output_root,
    )
    aggregate = aggregate_metrics(metrics)
    per_video = [asdict(metric) for metric in metrics]
    aggregate_row = asdict(aggregate)
    evaluation_contract = {
        "profile": PROFILE,
        "include_hidden": True,
        "iou_threshold": IOU_THRESHOLD,
        "gap_tolerance_frames": GAP_TOLERANCE_FRAMES,
        "identity_reporting": "RAW_AND_GLOBAL_REMAP_PER_VIDEO_THEN_AGGREGATE",
        "causal_timing_policy": "causal_framewise",
        "output_delay_frames": 0,
        "future_frames_used": False,
        "offline_repair": False,
        "post_video_smoothing": False,
    }
    gt_authority = [
        {
            "video_key": video.video_key,
            "gt_sha256": video.gt_sha256,
            "gt_authority": video.gt_authority,
        }
        for video in videos
    ]
    write_csv(
        output_root / "CURRENT_MAIN_PER_VIDEO_METRICS.csv",
        per_video,
    )
    write_json(
        output_root / "CURRENT_MAIN_BASELINE_METRICS.json",
        {
            "schema_version": "tracking.current_main_r0_metrics.v1",
            "aggregate": aggregate_row,
            "evaluation_contract": evaluation_contract,
        },
    )
    mp4_count = len(list(output_root.rglob("*.mp4")))
    if mp4_count:
        raise R0BaselineError("R0 run produced MP4")
    manifest = {
        "schema_version": "tracking.current_main_r0_run.v1",
        "r0_profile": PROFILE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": producer_sha,
        "tracking_tree_object": tracking_tree,
        "source_lineage_file_sha256": lineage_file_sha,
        "effective_config": config_payload,
        "effective_config_sha256": canonical_hash(config_payload),
        "detector_semantic_config": detector_payload,
        "detector_semantic_config_sha256": canonical_hash(detector_payload),
        "detector_weight_sha256": sha256_file(Path(cfg.weights_path)),
        "detector_cache_authority": cache_report,
        "gt_authority": gt_authority,
        "gt_authority_sha256": canonical_hash(gt_authority),
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": canonical_hash(evaluation_contract),
        "baseline_videos_total": 13,
        "baseline_videos_completed": len(metrics),
        "detector_inference_calls_during_tracking": 0,
        "cache_replay_calls_during_tracking": replay_calls,
        "h1_h2_validation_execution": False,
        "run_root_mp4_count": mp4_count,
        "current_main_error_taxonomy": "DEFERRED",
        "historical_reconciliation": "DEFERRED",
        "next_hypothesis_selection": "DEFERRED",
    }
    manifest_path = output_root / "CURRENT_MAIN_BASELINE_RUN_MANIFEST.json"
    write_json(manifest_path, manifest)
    write_json(
        output_root / "CURRENT_MAIN_BASELINE_REPRODUCIBILITY.json",
        {
            "schema_version": "tracking.current_main_r0_reproducibility.v1",
            "result": "PASS",
            "exact_source_gt_and_cache_hashes_verified": True,
            "cache_record_load_replay": "PASS",
            "detector_inference_calls_during_tracking": 0,
            "repeat_run": "NOT_RUN_NOT_REQUIRED_BY_FROZEN_POLICY",
            "run_root_mp4_count": 0,
        },
    )
    authority = {
        "schema_version": "tracking.current_main_r0_authority.v1",
        "r0_baseline_authority": "ESTABLISHED",
        "r0_profile": PROFILE,
        "r0_code_sha": producer_sha,
        "r0_run_root": str(output_root),
        "r0_manifest_sha256": sha256_file(manifest_path),
        "r0_config_sha256": manifest["effective_config_sha256"],
        "r0_detector_cache_authority_sha256": cache_report[
            "generation_decision_sha256"
        ],
        "r0_gt_authority_sha256": manifest["gt_authority_sha256"],
        "r0_evaluation_contract_sha256": manifest[
            "evaluation_contract_sha256"
        ],
        "r0_videos_completed": "13/13",
        "r0_per_video_metrics_ready": True,
        "aggregate_metrics": aggregate_row,
        "current_main_error_taxonomy": "DEFERRED",
        "historical_reconciliation": "DEFERRED",
        "next_hypothesis_selection": "DEFERRED",
    }
    write_json(
        output_root / "CURRENT_MAIN_R0_AUTHORITY.json",
        authority,
    )
    write_json(
        output_root / "CURRENT_MAIN_BASELINE_ENVIRONMENT.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "producer_code_sha": producer_sha,
        },
    )
    (output_root / "CURRENT_MAIN_BASELINE_COMMANDS.txt").write_text(
        subprocess.list2cmdline(sys.argv)
        + "\nDETECTOR_INFERENCE_CALLS_DURING_TRACKING=0\n"
        + "H1_H2_VALIDATION_EXECUTIONS=0\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "ARTIFACT_SHA256.json",
        {
            "schema_version": "tracking.current_main_r0_inventory.v1",
            "inventory_excludes_itself": True,
            "artifacts": inventory(output_root),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute(
        args.source_repo.resolve(),
        args.lineage_manifest.resolve(),
        args.cache_root.resolve(),
        args.output_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
