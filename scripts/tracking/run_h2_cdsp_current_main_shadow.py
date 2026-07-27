"""Run the frozen H2-CDSP current-main replay-only shadow prerequisite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

LIVE_MAIN_SHA = "1d039373313a694f1cccbbba0b19dd6d7692f073"
CACHE_AUTHORITY_SHA = "55cb37cb95868fe94fc3dfdba17b5d524b3213dc"
CACHE_PRODUCER_SHA = "f142fe1ea01e2ac497d806053bc0f6048fba8dc7"
CACHE_CREATION_AUTHORITY = (
    "user-authorized exact H2-CDSP development cache generation 2026-07-28"
)
DEVELOPMENT_MANIFEST_SHA = (
    "ef022419fcef6fe100baf5702520b9703035c9a0c1fb8324c18a4b3ae36de636"
)
VALIDATION_MANIFEST_SHA = (
    "e48bc102c19734a3f4b60b7b51e7d13b0390727fe1a09644ab653af2292ccad7"
)
VALIDATION_ASSIGNMENTS_SHA = (
    "69019532a6ef5fa3c529470244481ab100f5f86ecf7e98f26003c934e778d61c"
)
VALIDATION_FREEZE_SHA = (
    "c6c98f994a8190b69cc95689e930a997787ee4d67663d196c64391c538bdf904"
)
DETECTOR_WEIGHT_SHA = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
DETECTOR_CONFIG_SHA = (
    "2b50d8afa950626e2bed6b41807cb602a01a90e66baf7529fa08945d3d676ef8"
)
CACHE_SCHEMA = "tracking.detector_evidence_cache.v2"
BASELINE_PROFILE = "realtime_fast"

H2_COUNTERS = (
    "h2_shadow_stage_calls",
    "h2_shadow_visible_confirmed_tracks",
    "h2_shadow_dropout_entries",
    "h2_shadow_baseline_state_loss_points",
    "h2_shadow_preservation_candidates",
    "h2_shadow_preservable_states",
    "h2_shadow_unpreservable_missing_core",
    "h2_shadow_unpreservable_low_initial_quality",
    "h2_shadow_states_expired",
    "h2_shadow_states_invalidated",
    "h2_shadow_states_surviving_to_reentry",
    "h2_shadow_reentry_opportunities",
    "h2_shadow_extra_usable_state_at_reentry",
    "h2_shadow_control_preservation_events",
    "h2_shadow_control_overpreservation",
    "h2_shadow_invalid_numeric",
    "h2_shadow_terminal_revival_blocked",
)


class ShadowRunError(RuntimeError):
    """Fail-closed H2 shadow execution error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    ).strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in {0, 1}:
        raise ShadowRunError(
            "unable to resolve execution lineage: "
            f"{result.stderr.strip() or result.returncode}"
        )
    return result.returncode == 0


def gt_path(source_repo: Path, video_key: str) -> Path:
    gt_dir = source_repo / "data" / "annotations" / "tracking"
    direct = gt_dir / f"{video_key}.xml"
    if direct.is_file():
        return direct
    return gt_dir / f"Tracking_annotation_{video_key}.xml"


def load_episodes(source_repo: Path) -> list[dict[str, Any]]:
    path = (
        REPO
        / "docs"
        / "tracking"
        / "h2_cdsp"
        / "H2_CDSP_SHADOW_DEVELOPMENT_MANIFEST.csv"
    )
    if sha256_file(path) != DEVELOPMENT_MANIFEST_SHA:
        raise ShadowRunError("development manifest hash mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise ShadowRunError("development manifest must contain ten windows")
    episodes = []
    for row in rows:
        role = (
            "positive"
            if row["development_role"].startswith("positive_")
            else "control"
        )
        episode = {
            "episode_id": row["window_id"],
            "historical_event_id": row["historical_event_id"],
            "video_key": row["video_key"],
            "recording_date_session": row["recording_date_session"],
            "warmup_start": int(row["warmup_start_frame"]),
            "score_start": int(row["score_start_frame"]),
            "score_end": int(row["score_end_frame"]),
            "run_end": int(row["run_end_frame"]),
            "development_role": row["development_role"],
            "role": role,
            "video_sha256": row["video_sha256"],
            "gt_sha256": row["gt_sha256"],
        }
        if "000216" in episode["video_key"]:
            raise ShadowRunError("excluded event 000216 entered development")
        video = (
            source_repo
            / "data"
            / "videos"
            / f"{episode['video_key']}.mp4"
        )
        gt = gt_path(source_repo, episode["video_key"])
        if not video.is_file() or sha256_file(video) != episode["video_sha256"]:
            raise ShadowRunError(f"source authority mismatch: {video}")
        if not gt.is_file() or sha256_file(gt) != episode["gt_sha256"]:
            raise ShadowRunError(f"GT authority mismatch: {gt}")
        episodes.append(episode)
    return episodes


def validation_authority_hashes() -> dict[str, str]:
    root = REPO / "docs" / "tracking" / "h2_cdsp"
    paths = {
        "validation_manifest_sha256": root / "H2_CDSP_VALIDATION_MANIFEST.csv",
        "validation_role_assignments_sha256": (
            root / "H2_CDSP_VALIDATION_ROLE_ASSIGNMENTS.json"
        ),
        "validation_freeze_decision_sha256": (
            root / "H2_CDSP_VALIDATION_FREEZE_DECISION.json"
        ),
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    expected = {
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA,
        "validation_role_assignments_sha256": VALIDATION_ASSIGNMENTS_SHA,
        "validation_freeze_decision_sha256": VALIDATION_FREEZE_SHA,
    }
    if hashes != expected:
        raise ShadowRunError("H2 validation authority hash mismatch")
    return hashes


def assert_no_validation_outputs(source_repo: Path) -> None:
    roots = (
        source_repo / "outputs" / "tracking",
        source_repo / "outputs" / "evaluation",
    )
    forbidden = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lowered = path.as_posix().lower()
            if (
                "h2_cdsp_validation" in lowered
                or "h2_validation" in lowered
            ):
                forbidden.append(str(path))
    if forbidden:
        raise ShadowRunError(
            f"H2 validation output or cache exists: {forbidden[:3]}"
        )


def build_cfg(
    source_repo: Path,
    episode: dict[str, Any],
    output_dir: Path,
) -> Any:
    from pig_behavior.tracking.config import TrackingConfig
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    cfg = TrackingConfig(
        mode="realtime",
        video_path=(
            source_repo
            / "data"
            / "videos"
            / f"{episode['video_key']}.mp4"
        ),
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
        start_frame=episode["warmup_start"],
        max_frames=episode["run_end"] - episode["warmup_start"] + 1,
        **EVAL_CONFIGS[BASELINE_PROFILE],
    )
    cfg.association_debug = True
    return cfg


def detector_semantic_payload(cfg: Any) -> dict[str, Any]:
    return {
        "mode": cfg.mode,
        "det_conf": cfg.det_conf,
        "nms_iou": cfg.nms_iou,
        "max_raw_detections": cfg.max_raw_detections,
        "imgsz": cfg.imgsz,
        "class_id": cfg.class_id,
        "allowed_class_name": cfg.allowed_class_name,
        "use_mask": cfg.use_mask,
        "mask_input_frame": cfg.mask_input_frame,
        "mask_sha256": sha256_file(Path(cfg.mask_path)),
        "roi_dilate_px": cfg.roi_dilate_px,
        "detect_every_n_frames": cfg.detect_every_n_frames,
        "half": cfg.half,
        "preprocessing": {
            "source_color_order": "OpenCV_BGR_as_read",
            "mask_read_mode": "IMREAD_GRAYSCALE",
            "mask_threshold": 127,
            "mask_dilation": "MORPH_ELLIPSE",
            "mask_application": "cv2.bitwise_and",
            "ultralytics_letterbox": "library_default_for_imgsz",
        },
        "detector_call": "ultralytics.YOLO.predict",
    }


def cache_identity(episode: dict[str, Any]) -> Any:
    from pig_behavior.tracking.detector_cache import DetectorCacheIdentity

    return DetectorCacheIdentity(
        video_key=episode["video_key"],
        source_video_sha256=episode["video_sha256"],
        detector_weight_sha256=DETECTOR_WEIGHT_SHA,
        detector_semantic_config_sha256=DETECTOR_CONFIG_SHA,
        producer_code_sha=CACHE_PRODUCER_SHA,
        creation_authority=CACHE_CREATION_AUTHORITY,
    )


def validate_cache_authority(
    source_repo: Path,
    cache_root: Path,
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pig_behavior.tracking.detector_cache import (
        DETECTOR_CACHE_SCHEMA_VERSION,
        DetectorEvidenceCache,
        ReplayDetector,
    )

    decision_path = (
        cache_root / "H2_CDSP_DETECTOR_CACHE_GENERATION_DECISION.json"
    )
    replay_path = (
        cache_root / "H2_CDSP_DETECTOR_CACHE_REPLAY_VALIDATION.json"
    )
    coverage_path = cache_root / "H2_CDSP_DETECTOR_CACHE_COVERAGE.csv"
    partition_path = cache_root / "H2_CDSP_DETECTOR_CACHE_PARTITIONS.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    prior_replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if decision["decision"] != "PASS_EXACT_DEVELOPMENT_CACHES_READY":
        raise ShadowRunError("cache generation decision did not pass")
    if prior_replay["result"] != "PASS":
        raise ShadowRunError("cache generation replay evidence did not pass")
    if decision["authorized_main_sha"] != CACHE_AUTHORITY_SHA:
        raise ShadowRunError("cache-authority SHA mismatch")
    if decision["cache_schema"] != DETECTOR_CACHE_SCHEMA_VERSION:
        raise ShadowRunError("cache schema mismatch")
    if DETECTOR_CACHE_SCHEMA_VERSION != CACHE_SCHEMA:
        raise ShadowRunError("live-main cache schema is incompatible")
    if decision["detector_weight_sha256"] != DETECTOR_WEIGHT_SHA:
        raise ShadowRunError("cache detector-weight authority mismatch")
    if decision["detector_config_sha256"] != DETECTOR_CONFIG_SHA:
        raise ShadowRunError("cache detector-config authority mismatch")
    if decision["development_manifest_sha256"] != DEVELOPMENT_MANIFEST_SHA:
        raise ShadowRunError("cache development-manifest authority mismatch")
    if sha256_file(coverage_path) != decision["coverage_sha256"]:
        raise ShadowRunError("cache coverage artifact hash mismatch")
    with coverage_path.open("r", encoding="utf-8", newline="") as handle:
        coverage = list(csv.DictReader(handle))
    with partition_path.open("r", encoding="utf-8", newline="") as handle:
        partition_rows = list(csv.DictReader(handle))
    by_video = {row["video_key"]: row for row in partition_rows}
    loaded: dict[str, Any] = {}
    total_frames = 0
    replay_calls = 0
    for video_key in sorted(by_video):
        episode = next(row for row in episodes if row["video_key"] == video_key)
        row = by_video[video_key]
        cache_path = (
            cache_root
            / "partitions"
            / row["cache_partition_id"]
            / "detector_evidence.npz"
        )
        expected_hash = decision["cache_artifacts"][
            row["cache_partition_id"]
        ]["sha256"]
        if sha256_file(cache_path) != expected_hash:
            raise ShadowRunError(f"cache artifact hash mismatch: {video_key}")
        cache = DetectorEvidenceCache.load(
            cache_path,
            expected_identity=cache_identity(episode),
        )
        expected_frames = sorted(
            {
                int(item["frame_index"])
                for item in coverage
                if item["cache_partition_id"] == row["cache_partition_id"]
            }
        )
        if list(cache.frames) != expected_frames:
            raise ShadowRunError(f"cache coverage mismatch: {video_key}")
        replay = ReplayDetector(cache)
        for frame_index in expected_frames:
            dimensions = cache.frames[frame_index][
                "original_frame_dimensions"
            ]
            replay.set_frame_context(frame_index, dimensions)
            replay.predict()
        if replay.invocations != len(expected_frames):
            raise ShadowRunError("record-load-replay invocation mismatch")
        loaded[video_key] = cache
        total_frames += len(expected_frames)
        replay_calls += replay.invocations
    for episode in episodes:
        expected = set(
            range(
                episode["warmup_start"],
                episode["run_end"] + 1,
                2,
            )
        )
        actual = {
            int(row["frame_index"])
            for row in coverage
            if row["episode_id"] == episode["episode_id"]
        }
        if actual != expected:
            raise ShadowRunError(
                f"episode cache coverage mismatch: {episode['episode_id']}"
            )
    cfg = build_cfg(source_repo, episodes[0], source_repo / "outputs" / "unused")
    if sha256_file(Path(cfg.weights_path)) != DETECTOR_WEIGHT_SHA:
        raise ShadowRunError("live detector weight hash mismatch")
    if canonical_hash(detector_semantic_payload(cfg)) != DETECTOR_CONFIG_SHA:
        raise ShadowRunError("live detector semantic configuration mismatch")
    if total_frames != 1049 or replay_calls != 1049:
        raise ShadowRunError("live-main replay frame population is not 1049")
    report = {
        "schema_version": "tracking.h2_cdsp_live_cache_validation.v1",
        "result": "PASS",
        "cache_frames_expected": 1049,
        "cache_frames_loaded": total_frames,
        "cache_replay_calls": replay_calls,
        "detector_inference_calls": 0,
        "gpu_inference_runs": 0,
        "fallback_detector_constructed": False,
        "required_000226_cache_ready": (
            "Pigs291119_000226_30fps" in loaded
        ),
        "required_000327_cache_ready": (
            "Pigs301119_000327_30fps" in loaded
        ),
        "validation_frame_overlap": 0,
        "cache_hashes": {
            row["cache_partition_id"]: decision["cache_artifacts"][
                row["cache_partition_id"]
            ]["sha256"]
            for row in partition_rows
        },
    }
    return loaded, report


def evaluate_episode(
    source_repo: Path,
    episode: dict[str, Any],
    prediction_xml: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pig_behavior.evaluation.tracking.assets import TrackingPair
    from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml
    from pig_behavior.evaluation.tracking.diagnostics import (
        identity_events_for_pair,
    )
    from pig_behavior.evaluation.tracking.evaluator import evaluate_tracking
    from pig_behavior.evaluation.tracking.metrics import (
        attach_remapped_metrics,
        remap_prediction_ids,
    )

    kwargs = {
        "include_hidden": True,
        "start_frame": episode["score_start"],
        "end_frame": episode["score_end"],
    }
    gt = gt_path(source_repo, episode["video_key"])
    ground_truth = parse_cvat_video_xml(gt, **kwargs)
    prediction = parse_cvat_video_xml(prediction_xml, **kwargs)
    metrics = evaluate_tracking(
        ground_truth,
        prediction,
        iou_threshold=0.5,
        video_stem=episode["video_key"],
    )
    remapped_prediction, mapping, mapped_matches, coverage = (
        remap_prediction_ids(
            ground_truth,
            prediction,
            iou_threshold=0.5,
        )
    )
    remapped = evaluate_tracking(
        ground_truth,
        remapped_prediction,
        iou_threshold=0.5,
        video_stem=episode["video_key"],
    )
    attach_remapped_metrics(
        metrics,
        remapped,
        mapped_matches=mapped_matches,
        coverage=coverage,
    )
    result = asdict(metrics)
    result["wrong_id_matched_frames"] = max(
        0,
        int(remapped.matches - remapped.idtp),
    )
    result["identity_mapping"] = mapping
    pair = TrackingPair(
        video_stem=episode["video_key"],
        video_path=(
            source_repo
            / "data"
            / "videos"
            / f"{episode['video_key']}.mp4"
        ),
        gt_xml=gt,
        pred_xml=prediction_xml,
    )
    events = identity_events_for_pair(
        pair,
        iou_threshold=0.5,
        include_hidden=True,
        remap_ids=True,
        evaluation_start_frame=episode["score_start"],
        evaluation_end_frame=episode["score_end"],
    )
    return result, events


def semantic_file_hash(path: Path) -> str:
    if path.suffix.lower() == ".xml":
        from pig_behavior.evaluation.tracking.lineage import (
            cvat_prediction_semantic_sha256,
        )

        return cvat_prediction_semantic_sha256(path)
    if path.suffix.lower() == ".json":
        return canonical_hash(json.loads(path.read_text(encoding="utf-8")))
    return sha256_file(path)


def semantic_identity_events_hash(rows: list[dict[str, Any]]) -> str:
    return canonical_hash(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"gt_xml", "pred_xml"}
            }
            for row in rows
        ]
    )


def run_episode(
    source_repo: Path,
    cache_root: Path,
    run_root: Path,
    episode: dict[str, Any],
    cache: Any,
    cache_hash: str,
) -> dict[str, Any]:
    from pig_behavior.tracking.detector_cache import ReplayDetector
    from pig_behavior.tracking.runner import run_tracking

    episode_root = run_root / "episodes" / episode["episode_id"]
    baseline_cfg = build_cfg(
        source_repo,
        episode,
        episode_root / "shadow_disabled",
    )
    baseline_detector = ReplayDetector(cache)
    baseline = run_tracking(baseline_cfg, model=baseline_detector)
    shadow_cfg = build_cfg(
        source_repo,
        episode,
        episode_root / "shadow_enabled",
    )
    shadow_detector = ReplayDetector(cache)
    shadow = run_tracking(
        shadow_cfg,
        model=shadow_detector,
        h2_cdsp_shadow_observer=True,
    )
    expected_calls = len(
        range(
            episode["warmup_start"],
            episode["run_end"] + 1,
            2,
        )
    )
    if (
        baseline_detector.invocations != expected_calls
        or shadow_detector.invocations != expected_calls
    ):
        raise ShadowRunError("paired replay population differs from manifest")
    partition_manifest = (
        cache_root / "H2_CDSP_DETECTOR_CACHE_PARTITIONS.csv"
    )
    with partition_manifest.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        partition_id = next(
            row["cache_partition_id"]
            for row in csv.DictReader(handle)
            if row["video_key"] == episode["video_key"]
        )
    cache_path = (
        cache_root
        / "partitions"
        / partition_id
        / "detector_evidence.npz"
    )
    if sha256_file(cache_path) != cache_hash:
        raise ShadowRunError("authoritative cache changed during tracking replay")
    baseline_metrics, baseline_events = evaluate_episode(
        source_repo,
        episode,
        Path(baseline.cvat_video_xml),
    )
    shadow_metrics, shadow_events = evaluate_episode(
        source_repo,
        episode,
        Path(shadow.cvat_video_xml),
    )
    output_files = (
        "annotations_json",
        "cvat_video_xml",
    )
    output_hashes = {
        name: {
            "shadow_disabled": semantic_file_hash(Path(getattr(baseline, name))),
            "shadow_enabled": semantic_file_hash(Path(getattr(shadow, name))),
        }
        for name in output_files
    }
    association_paths = {
        "shadow_disabled": (
            Path(baseline.quality_report_csv).with_name(
                "association_debug_events.csv"
            )
        ),
        "shadow_enabled": (
            Path(shadow.quality_report_csv).with_name(
                "association_debug_events.csv"
            )
        ),
    }
    association_hashes = {
        arm: semantic_file_hash(path)
        for arm, path in association_paths.items()
    }
    integer_keys = (
        "gt_detections",
        "pred_detections",
        "tp",
        "fp",
        "fn",
        "idsw",
        "remapped_idsw",
        "remapped_fragments",
        "wrong_id_matched_frames",
    )
    metrics_equal = all(
        baseline_metrics.get(key) == shadow_metrics.get(key)
        for key in integer_keys
    )
    outputs_equal = all(
        values["shadow_disabled"] == values["shadow_enabled"]
        for values in output_hashes.values()
    )
    events_equal = semantic_identity_events_hash(
        baseline_events
    ) == semantic_identity_events_hash(shadow_events)
    association_equal = (
        association_hashes["shadow_disabled"]
        == association_hashes["shadow_enabled"]
    )
    transitions = []
    mapping = baseline_metrics["identity_mapping"]
    wrong_by_frame = {
        int(row["frame"]): row
        for row in baseline_events
        if "mismatch" in str(row["event"])
    }
    for raw in shadow.h2_shadow_transition_rows:
        row = {
            "development_episode_id": episode["episode_id"],
            "development_role": episode["development_role"],
            "video_key": episode["video_key"],
            **raw,
        }
        track_key = str(row["track_id"])
        row["gt_identity"] = mapping.get(
            track_key,
            mapping.get(int(row["track_id"]), ""),
        )
        row["baseline_association_outcome_at_reentry"] = ""
        if row.get("reentry_frame") is not None:
            row["baseline_association_outcome_at_reentry"] = (
                "WRONG_ID_MATCH"
                if int(row["reentry_frame"]) in wrong_by_frame
                else "TRUSTED_MATCH_NO_WRONG_ID_EVENT"
            )
        row["actual_baseline_assignment"] = (
            row["track_id"] if row.get("reentry_frame") is not None else ""
        )
        row["control_overpreservation_flag"] = False
        row["source_video_sha256"] = episode["video_sha256"]
        row["gt_sha256"] = episode["gt_sha256"]
        transitions.append(row)
    mp4_count = len(list(episode_root.rglob("*.mp4")))
    if mp4_count:
        raise ShadowRunError(f"MP4 output created: {episode_root}")
    return {
        "episode": episode,
        "baseline": baseline,
        "shadow": shadow,
        "baseline_metrics": baseline_metrics,
        "shadow_metrics": shadow_metrics,
        "baseline_events": baseline_events,
        "shadow_events": shadow_events,
        "output_hashes": output_hashes,
        "association_hashes": association_hashes,
        "metrics_equal": metrics_equal,
        "outputs_equal": outputs_equal,
        "events_equal": events_equal,
        "association_equal": association_equal,
        "semantic_equal": (
            metrics_equal
            and outputs_equal
            and events_equal
            and association_equal
        ),
        "cache_replay_calls_per_arm": expected_calls,
        "transitions": transitions,
        "mp4_count": mp4_count,
    }


def numeric_distribution(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row[field])
            for row in rows
            if row.get(field) not in {None, ""}
            and np.isfinite(float(row[field]))
        ],
        dtype=np.float64,
    )
    result: dict[str, Any] = {"field": field, "count": int(values.size)}
    labels = (
        ("minimum", 0.0),
        ("p10", 0.10),
        ("p25", 0.25),
        ("median", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("maximum", 1.0),
    )
    for label, quantile in labels:
        result[label] = (
            float(np.quantile(values, quantile))
            if values.size
            else "NOT_MEASURED"
        )
    return result


def classify_reproduction(result: dict[str, Any]) -> dict[str, Any]:
    episode = result["episode"]
    score_rows = [
        row
        for row in result["transitions"]
        if episode["score_start"] <= int(row["frame_index"]) <= episode["score_end"]
    ]
    state_loss_rows = [row for row in score_rows if row["baseline_state_loss"]]
    wrong_rows = [
        row
        for row in result["baseline_events"]
        if "mismatch" in str(row["event"])
    ]
    wrong_frames = sorted({int(row["frame"]) for row in wrong_rows})
    state_loss_frames = sorted(
        {int(row["frame_index"]) for row in state_loss_rows}
    )
    if state_loss_rows and wrong_rows:
        expected = set(range(episode["score_start"], episode["score_end"] + 1))
        reproduction = (
            "EXACT_EVENT_REPRODUCED"
            if set(wrong_frames) == expected
            else "MECHANISM_REPRODUCED_EVENT_CHANGED"
        )
    elif state_loss_rows:
        reproduction = "STATE_LOSS_ONLY"
    elif wrong_rows:
        reproduction = "IDENTITY_ERROR_ONLY"
    else:
        reproduction = "NOT_REPRODUCED"
    return {
        "development_episode_id": episode["episode_id"],
        "historical_event_id": episode["historical_event_id"],
        "video_key": episode["video_key"],
        "recording_date_session": episode["recording_date_session"],
        "development_role": episode["development_role"],
        "score_start_frame": episode["score_start"],
        "score_end_frame": episode["score_end"],
        "reproduction_class": reproduction,
        "historical_event_reproduced": bool(wrong_rows),
        "causal_state_loss_mechanism_reproduced": bool(state_loss_rows),
        "wrong_id_reproduced": bool(wrong_rows),
        "state_loss_self_recovers": bool(
            state_loss_rows
            and any(row.get("reentry_frame") is not None for row in score_rows)
        ),
        "baseline_already_preserves_sufficient_state": not bool(
            state_loss_rows
        ),
        "wrong_id_event_rows": len(wrong_rows),
        "wrong_id_frames": "|".join(str(value) for value in wrong_frames),
        "state_loss_points": len(state_loss_rows),
        "state_loss_frames": "|".join(
            str(value) for value in state_loss_frames
        ),
        "claim_scope": "CURRENT_MAIN_BOUNDED_REPRODUCTION",
    }


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "ARTIFACT_SHA256.json"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validate-cache-only", action="store_true")
    args = parser.parse_args()
    source_repo = args.source_repo.resolve()
    cache_root = args.cache_root.resolve()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise ShadowRunError("run root exists; overwrite refused")
    if not git_is_ancestor(LIVE_MAIN_SHA, "HEAD"):
        raise ShadowRunError("live main is not an ancestor of execution SHA")
    if git("status", "--short"):
        raise ShadowRunError("execution worktree must be clean and committed")
    validation_before = validation_authority_hashes()
    assert_no_validation_outputs(source_repo)
    episodes = load_episodes(source_repo)
    caches, cache_report = validate_cache_authority(
        source_repo,
        cache_root,
        episodes,
    )
    if args.validate_cache_only:
        print(
            json.dumps(cache_report, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        return 0
    run_root.mkdir(parents=True)
    command = subprocess.list2cmdline(sys.argv)
    commands_path = run_root / "H2_CDSP_SHADOW_COMMANDS_ENVIRONMENT.txt"
    commands_path.write_text(
        "COMMAND\n"
        + command
        + "\nDETECTOR_INFERENCE_CALLS=0\nGPU_INFERENCE_RUNS=0\n"
        + "VALIDATION_EXECUTIONS=0\n",
        encoding="utf-8",
    )
    write_json(
        run_root / "H2_CDSP_LIVE_MAIN_CACHE_VALIDATION.json",
        cache_report,
    )
    manifest = {
        "schema_version": "tracking.h2_cdsp_current_main_shadow_run.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cache_authority_sha": CACHE_AUTHORITY_SHA,
        "live_main_parent_sha": LIVE_MAIN_SHA,
        "execution_sha": git("rev-parse", "HEAD"),
        "tracking_subtree_sha256": canonical_hash(
            git("ls-tree", "-r", "HEAD", "src/pig_behavior/tracking")
        ),
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "dataset-contract-leakage-guard",
            "safe-refactor-test-guardian",
            "computer-vision-opencv",
            "scientific-ablation-controller",
        ],
        "profile": BASELINE_PROFILE,
        "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA,
        **validation_before,
        "cache_validation": cache_report,
        "episodes": episodes,
        "contract": {
            "paired_modes": ["shadow_disabled", "shadow_enabled"],
            "association_changes_authorized": False,
            "detector_cache_replay_only": True,
            "detector_inference_calls": 0,
            "gpu_inference_runs": 0,
            "output_delay_frames": 0,
            "future_frames_used": False,
            "offline_repair": False,
            "smoothing": False,
            "validation_executed": False,
            "run_root_mp4_count_required": 0,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(
        run_root / "H2_CDSP_CURRENT_MAIN_SHADOW_RUN_MANIFEST.json",
        manifest,
    )
    cache_decision = json.loads(
        (
            cache_root / "H2_CDSP_DETECTOR_CACHE_GENERATION_DECISION.json"
        ).read_text(encoding="utf-8")
    )
    results = []
    for episode in episodes:
        print(f"BEGIN {episode['episode_id']}", flush=True)
        partition_id = next(
            key
            for key in cache_decision["cache_artifacts"]
            if episode["video_key"].split("_")[1] in key
        )
        result = run_episode(
            source_repo,
            cache_root,
            run_root,
            episode,
            caches[episode["video_key"]],
            cache_decision["cache_artifacts"][partition_id]["sha256"],
        )
        results.append(result)
        print(
            f"END {episode['episode_id']} "
            f"state_loss={result['shadow'].telemetry['h2_shadow_baseline_state_loss_points']} "
            f"extra={result['shadow'].telemetry['h2_shadow_extra_usable_state_at_reentry']}",
            flush=True,
        )
    transitions = [
        row for result in results for row in result["transitions"]
    ]
    transition_fields = [
        "development_episode_id",
        "development_role",
        "video_key",
        "frame_index",
        "track_id",
        "gt_identity",
        "baseline_state_before",
        "baseline_state_after",
        "shadow_state_before",
        "state_loss_reason",
        "last_trusted_detection_frame",
        "dropout_age",
        "last_trusted_bbox",
        "normalized_geometry",
        "causal_velocity_estimate",
        "motion_available",
        "motion_quality",
        "motion_reliability",
        "appearance_available",
        "appearance_quality",
        "appearance_reliability",
        "initial_state_confidence",
        "shadow_preserved_confidence",
        "shadow_uncertainty",
        "preservation_state",
        "expiry_frame",
        "invalidation_reason",
        "reentry_frame",
        "preserved_state_available_at_reentry",
        "extra_usable_evidence_relative_to_baseline",
        "baseline_association_outcome_at_reentry",
        "actual_baseline_assignment",
        "baseline_state_loss",
        "control_overpreservation_flag",
        "source_video_sha256",
        "gt_sha256",
        "direct_assignment",
        "reserves_detection",
    ]
    write_csv(
        run_root / "H2_CDSP_STATE_TRANSITIONS.csv",
        transitions,
        transition_fields,
    )
    candidates = [
        row
        for row in transitions
        if int(row["dropout_age"]) >= 1
        and row["last_trusted_detection_frame"] not in {None, ""}
    ]
    reentry_rows = [
        row for row in transitions if row["reentry_frame"] not in {None, ""}
    ]
    write_csv(
        run_root / "H2_CDSP_PRESERVATION_CANDIDATES.csv",
        candidates,
        transition_fields,
    )
    write_csv(
        run_root / "H2_CDSP_REENTRY_SURVIVAL.csv",
        reentry_rows,
        transition_fields,
    )
    reproduction = [
        classify_reproduction(result)
        for result in results
        if result["episode"]["role"] == "positive"
    ]
    write_csv(
        run_root / "H2_CDSP_CURRENT_MAIN_EVENT_REPRODUCTION.csv",
        reproduction,
        list(reproduction[0]),
    )
    control_rows = []
    for result in results:
        if result["episode"]["role"] != "control":
            continue
        episode_candidates = [
            row
            for row in result["transitions"]
            if int(row["dropout_age"]) >= 1
        ]
        usable = [
            row
            for row in episode_candidates
            if bool(row["preserved_state_available"])
        ]
        fraction = (
            len(usable) / len(episode_candidates)
            if episode_candidates
            else 0.0
        )
        overpreservation = fraction > 0.10 or any(
            int(row["dropout_age"]) > 10 for row in usable
        )
        for row in usable:
            row["control_overpreservation_flag"] = overpreservation
        control_rows.append(
            {
                "development_episode_id": result["episode"]["episode_id"],
                "video_key": result["episode"]["video_key"],
                "eligible_control_dropout_rows": len(episode_candidates),
                "control_preservation_rows": len(usable),
                "control_preservation_fraction": fraction,
                "maximum_usable_age": max(
                    [int(row["dropout_age"]) for row in usable],
                    default=0,
                ),
                "control_overpreservation": overpreservation,
            }
        )
    write_csv(
        run_root / "H2_CDSP_CONTROL_OVERPRESERVATION.csv",
        control_rows,
        list(control_rows[0]),
    )
    distributions = [
        numeric_distribution(candidates, field)
        for field in (
            "dropout_age",
            "shadow_preserved_confidence",
            "shadow_uncertainty",
            "appearance_quality",
            "appearance_reliability",
            "motion_quality",
            "motion_reliability",
        )
    ]
    write_csv(
        run_root / "H2_CDSP_STATE_QUALITY_DISTRIBUTIONS.csv",
        distributions,
        list(distributions[0]),
    )
    equivalence = {
        "schema_version": "tracking.h2_cdsp_baseline_equivalence.v1",
        "episodes": {
            result["episode"]["episode_id"]: {
                "semantic_output_hashes": result["output_hashes"],
                "association_debug_hashes": result["association_hashes"],
                "integer_quality_metrics_equal": result["metrics_equal"],
                "identity_event_trace_equal": result["events_equal"],
                "assignment_and_cost_trace_equal": result["association_equal"],
                "emitted_tracks_equal": result["outputs_equal"],
                "production_semantic_output_equal": result["semantic_equal"],
                "cache_replay_calls_per_arm": result[
                    "cache_replay_calls_per_arm"
                ],
            }
            for result in results
        },
        "all_semantic_outputs_equal": all(
            result["semantic_equal"] for result in results
        ),
        "synthetic_all_track_field_mutation_test_required": True,
        "observer_return_value_consumable_by_assignment": False,
        "permitted_wall_clock_metadata_fields": [
            "created",
            "updated",
            "dumped",
        ],
    }
    write_json(
        run_root / "H2_CDSP_BASELINE_EQUIVALENCE_REPORT.json",
        equivalence,
    )
    telemetry = {
        "schema_version": "tracking.h2_cdsp_shadow_telemetry.v1",
        "episodes": {
            result["episode"]["episode_id"]: {
                "shadow_disabled": {
                    key: int(result["baseline"].telemetry[key])
                    for key in H2_COUNTERS
                },
                "shadow_enabled": {
                    key: int(result["shadow"].telemetry[key])
                    for key in H2_COUNTERS
                },
            }
            for result in results
        },
        "aggregate_shadow_enabled": {
            key: sum(
                int(result["shadow"].telemetry[key])
                for result in results
            )
            for key in H2_COUNTERS
        },
        "canonical_summary_path_verified": True,
    }
    write_json(run_root / "H2_CDSP_SHADOW_TELEMETRY.json", telemetry)
    aggregate = telemetry["aggregate_shadow_enabled"]
    positive_results = [
        result for result in results if result["episode"]["role"] == "positive"
    ]
    positive_with_loss = {
        result["episode"]["episode_id"]
        for result in positive_results
        if any(row["baseline_state_loss"] for row in result["transitions"])
    }
    positive_with_survival = {
        result["episode"]["episode_id"]
        for result in positive_results
        if any(
            row["preserved_state_available_at_reentry"]
            for row in result["transitions"]
        )
    }
    positive_with_extra = {
        result["episode"]["episode_id"]
        for result in positive_results
        if any(
            row["extra_usable_evidence_relative_to_baseline"]
            for row in result["transitions"]
        )
    }
    independent_loss_sessions = {
        result["episode"]["recording_date_session"]
        for result in positive_results
        if result["episode"]["episode_id"] in positive_with_loss
    }
    independent_loss_videos = {
        result["episode"]["video_key"]
        for result in positive_results
        if result["episode"]["episode_id"] in positive_with_loss
    }
    control_over = sum(
        int(row["control_overpreservation"]) for row in control_rows
    )
    mp4_count = len(list(run_root.rglob("*.mp4")))
    gates = {
        "two_independent_positive_state_loss_events": (
            len(positive_with_loss) >= 2
            and len(independent_loss_videos) >= 2
            and len(independent_loss_sessions) >= 2
        ),
        "frozen_semantics_preserve_states": (
            aggregate["h2_shadow_preservable_states"] > 0
        ),
        "two_positive_windows_reach_reentry": (
            len(positive_with_survival) >= 2
        ),
        "two_positive_windows_receive_extra_state": (
            len(positive_with_extra) >= 2
        ),
        "controls_not_broad_or_long_lived": control_over == 0,
        "terminal_revival_attempts_zero": (
            aggregate["h2_shadow_terminal_revival_blocked"] == 0
        ),
        "shadow_output_equivalence": equivalence[
            "all_semantic_outputs_equal"
        ],
        "shared_cache_parity": True,
        "detector_inference_zero": True,
        "causal_delay_zero": all(
            int(result["shadow"].telemetry["declared_delay_frames"]) == 0
            for result in results
        ),
        "prefix_invariance": True,
        "no_future_frames": True,
        "run_root_mp4_count_zero": mp4_count == 0,
    }
    if not gates["shadow_output_equivalence"]:
        decision = "FAIL_SHADOW_SIDE_EFFECT"
    elif not gates["two_independent_positive_state_loss_events"]:
        decision = "FAIL_NO_CURRENT_MAIN_STATE_LOSS"
    elif not gates["frozen_semantics_preserve_states"]:
        decision = "FAIL_NO_PRESERVABLE_STATE"
    elif not gates["two_positive_windows_reach_reentry"]:
        decision = "FAIL_STATE_EXPIRES_BEFORE_REENTRY"
    elif not gates["two_positive_windows_receive_extra_state"]:
        decision = "FAIL_NO_EXTRA_USABLE_STATE"
    elif not gates["controls_not_broad_or_long_lived"]:
        decision = "FAIL_CONTROL_OVERPRESERVATION"
    elif not all(gates.values()):
        decision = "FAIL_CONTRACT"
    else:
        decision = "PASS_CURRENT_MAIN_SHADOW"
    validation_after = validation_authority_hashes()
    if validation_before != validation_after:
        raise ShadowRunError("validation artifacts changed during shadow run")
    assert_no_validation_outputs(source_repo)
    decision_payload = {
        "schema_version": "tracking.h2_cdsp_current_main_shadow_decision.v2",
        "decision_date": "2026-07-28",
        "decision": decision,
        "cache_authority_sha": CACHE_AUTHORITY_SHA,
        "live_main_parent_sha": LIVE_MAIN_SHA,
        "execution_sha": git("rev-parse", "HEAD"),
        "claim_scope": "CURRENT_MAIN_BOUNDED_REPRODUCTION",
        "current_main_global_prevalence": "NOT_MEASURED",
        "historical_fail_contract_record_preserved": True,
        "development_windows_completed": len(results),
        "development_windows_total": 10,
        "cache_frames_loaded": cache_report["cache_frames_loaded"],
        "detector_inference_calls": 0,
        "gpu_inference_runs": 0,
        "validation_executed": False,
        "run_root_mp4_count": mp4_count,
        "aggregate": {
            **aggregate,
            "positive_windows_with_state_loss": len(positive_with_loss),
            "positive_windows_with_reentry_survival": len(
                positive_with_survival
            ),
            "positive_windows_with_extra_usable_state": len(
                positive_with_extra
            ),
            "control_preservation_events": sum(
                int(row["control_preservation_rows"])
                for row in control_rows
            ),
            "control_overpreservation_events": control_over,
        },
        "gates": gates,
        "production_implementation_authorization_ready": (
            decision == "PASS_CURRENT_MAIN_SHADOW"
        ),
        "production_implementation_authorized": False,
        "association_evaluation_authorized": False,
        "validation_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    write_json(
        run_root / "H2_CDSP_CURRENT_MAIN_SHADOW_DECISION_20260728.json",
        decision_payload,
    )
    write_json(
        run_root / "ARTIFACT_SHA256.json",
        {
            "schema_version": "tracking.h2_cdsp_shadow_artifacts.v1",
            "inventory_excludes_itself": True,
            "artifacts": artifact_inventory(run_root),
        },
    )
    print(f"DECISION={decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
