"""Run and audit the exact-current-main replay-only full-13 tracking baseline."""

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

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "tracking"))

from build_rf_acc23_error_taxonomy import (  # noqa: E402
    _window_debug_evidence,
    _window_frame_evidence,
    group_error_events,
)
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
PERMANENT_DURATION_FRAMES = 60
HISTORICAL_TAXONOMY = (
    REPO.parent
    / "PIG_Behavior_Project"
    / "outputs"
    / "tracking"
    / "rf_acc23_error_taxonomy_20260727"
    / "identity_error_events.csv"
)
MECHANISMS = (
    "DETECTION_MISS_OR_DROPOUT",
    "DETECTION_MERGE_OR_SPLIT",
    "VISIBLE_VISIBLE_ASSOCIATION_AMBIGUITY",
    "OCCLUSION_OWNER_LOSS",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION",
    "APPEARANCE_DRIFT_OR_UNAVAILABLE",
    "MOTION_PROPAGATION_FAILURE",
    "TRACK_BIRTH_OR_DUPLICATE_TRACK",
    "TRACK_TERMINATION_OR_REVIVAL_POLICY",
    "SHORT_TRANSIENT_IDENTITY_ERROR",
    "GT_OR_EVALUATION_AMBIGUITY",
    "OTHER_MEASURED",
    "UNRESOLVED",
)
H1_TARGETS = {
    "VISIBLE_VISIBLE_ASSOCIATION_AMBIGUITY": "PARTIAL",
    "OCCLUSION_OWNER_LOSS": "YES",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION": "PARTIAL",
}
H2_TARGETS = {
    "DETECTION_MISS_OR_DROPOUT": "PARTIAL",
    "OCCLUSION_OWNER_LOSS": "YES",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION": "YES",
    "TRACK_TERMINATION_OR_REVIVAL_POLICY": "PARTIAL",
}


class BaselineAuditError(RuntimeError):
    """Fail-closed baseline replay or scientific audit error."""


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_MAIN_SHA, head],
        cwd=REPO,
        check=False,
    ).returncode:
        raise BaselineAuditError("starting-main authority is not an ancestor")
    starting_tree = git_output(
        "rev-parse",
        f"{STARTING_MAIN_SHA}:src/pig_behavior/tracking",
    )
    current_tree = git_output("rev-parse", "HEAD:src/pig_behavior/tracking")
    if current_tree != starting_tree:
        raise BaselineAuditError("tracking subtree differs from starting main")
    if git_output("status", "--short"):
        raise BaselineAuditError("baseline producer worktree must be clean")
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
        raise BaselineAuditError("cache-generation decision did not pass")
    if replay_authority["result"] != "PASS":
        raise BaselineAuditError("cache replay authority did not pass")
    if decision["cache_schema"] != DETECTOR_CACHE_SCHEMA_VERSION:
        raise BaselineAuditError("cache schema differs from live main")
    caches: dict[str, Any] = {}
    loaded_frames = 0
    cache_hashes: dict[str, str] = {}
    for video in videos:
        path = cache_path(cache_root, video.video_key)
        expected = decision["cache_artifacts"][video.video_key]["sha256"]
        if sha256_file(path) != expected:
            raise BaselineAuditError(f"cache hash mismatch: {video.video_key}")
        cache = DetectorEvidenceCache.load(
            path,
            expected_identity=cache_identity(video, decision),
        )
        expected_frames = frame_indices(video, 2)
        if tuple(cache.frames) != expected_frames:
            raise BaselineAuditError(
                f"cache frame coverage mismatch: {video.video_key}"
            )
        replay = ReplayDetector(cache)
        for frame_index in expected_frames:
            replay.set_frame_context(
                frame_index,
                cache.frames[frame_index]["original_frame_dimensions"],
            )
            replay.predict()
        if replay.invocations != 900:
            raise BaselineAuditError("record-load-replay count mismatch")
        caches[video.video_key] = cache
        cache_hashes[video.video_key] = expected
        loaded_frames += replay.invocations
    if loaded_frames != 11700:
        raise BaselineAuditError("full-13 cache population is not 11,700")
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
    cfg.association_debug = True
    validate_config(cfg)
    return cfg


def evaluate_video(
    video: VideoAuthority,
    prediction_xml: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, str]]:
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

    ground_truth = parse_cvat_video_xml(video.gt_path, include_hidden=True)
    prediction = parse_cvat_video_xml(prediction_xml, include_hidden=True)
    metrics = evaluate_tracking(
        ground_truth,
        prediction,
        iou_threshold=IOU_THRESHOLD,
        video_stem=video.video_key,
        gap_tolerance_frames=GAP_TOLERANCE_FRAMES,
    )
    remapped_prediction, mapping, mapped_matches, coverage = (
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
    pair = TrackingPair(
        video_stem=video.video_key,
        video_path=video.video_path,
        gt_xml=video.gt_path,
        pred_xml=prediction_xml,
    )
    rows = identity_events_for_pair(
        pair,
        iou_threshold=IOU_THRESHOLD,
        include_hidden=True,
        remap_ids=True,
    )
    return metrics, rows, mapping


def run_predictions(
    source_repo: Path,
    videos: list[VideoAuthority],
    caches: dict[str, Any],
    output_root: Path,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, dict[str, str]], int]:
    from pig_behavior.tracking.detector_cache import ReplayDetector
    from pig_behavior.tracking.runner import run_tracking

    metrics = []
    identity_rows: list[dict[str, Any]] = []
    mappings: dict[str, dict[str, str]] = {}
    replay_calls = 0
    for video in videos:
        print(f"BASELINE_BEGIN {video.video_key}", flush=True)
        output_dir = output_root / "predictions" / video.video_key
        cfg = build_cfg(source_repo, video, output_dir)
        detector = ReplayDetector(caches[video.video_key])
        summary = run_tracking(cfg, model=detector)
        if detector.invocations != 900:
            raise BaselineAuditError(
                f"tracking replay population mismatch: {video.video_key}"
            )
        metric, rows, mapping = evaluate_video(
            video,
            Path(summary.cvat_video_xml),
        )
        for row in rows:
            row["source_video_sha256"] = video.video_sha256
            row["gt_sha256"] = video.gt_sha256
            row["gt_authority"] = video.gt_authority
        metrics.append(metric)
        identity_rows.extend(rows)
        mappings[video.video_key] = mapping
        replay_calls += detector.invocations
        print(
            f"BASELINE_END {video.video_key} "
            f"idsw={metric.remapped_idsw} rows={len(rows)}",
            flush=True,
        )
    if replay_calls != 11700:
        raise BaselineAuditError("tracking did not replay all 11,700 frames")
    return metrics, identity_rows, mappings, replay_calls


def classify_event(event: dict[str, Any]) -> tuple[str, str]:
    secondary: list[str] = []
    if event["gt_authority"] != "AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS":
        return "GT_OR_EVALUATION_AMBIGUITY", ""
    if event["start_frame"] == 0:
        return "TRACK_BIRTH_OR_DUPLICATE_TRACK", ""
    if event["duration_frames"] <= 2:
        return "SHORT_TRANSIENT_IDENTITY_ERROR", ""
    if event["hidden_count_max"] and event["overlap_pair_count_max"]:
        if event["missing_detection_count_max"]:
            secondary.append("DETECTION_MISS_OR_DROPOUT")
        return "OCCLUSION_OWNER_LOSS", "|".join(secondary)
    if event["reid_phase_present"] and event["lost_track_count_max"]:
        return "REENTRY_AFTER_LONG_HIDDEN_DURATION", ""
    if event["missing_detection_count_max"]:
        return "DETECTION_MISS_OR_DROPOUT", ""
    if event["overlap_pair_count_max"]:
        return "VISIBLE_VISIBLE_ASSOCIATION_AMBIGUITY", ""
    if event["low_conf_recovery_present"] and event["max_track_missed_at_onset"] != (
        "NOT_EXPORTED"
    ):
        return "TRACK_TERMINATION_OR_REVIVAL_POLICY", ""
    if event["appearance_availability"] == "NOT_EXPORTED":
        secondary.append("APPEARANCE_DRIFT_OR_UNAVAILABLE")
    return "UNRESOLVED", "|".join(secondary)


def enrich_events(
    raw_events: list[dict[str, Any]],
    output_root: Path,
    videos: dict[str, VideoAuthority],
) -> list[dict[str, Any]]:
    cache: dict[str, tuple[dict[str, Any], pd.DataFrame, dict[str, str]]] = {}
    events = []
    for index, event in enumerate(raw_events, start=1):
        video_key = event["video_key"]
        if video_key not in cache:
            prediction_root = output_root / "predictions" / video_key
            quality = prediction_root / "tracking_quality_report.json"
            debug = prediction_root / "association_debug_events.csv"
            prediction = prediction_root / "annotations_cvat_video_1_1.xml"
            cache[video_key] = (
                json.loads(quality.read_text(encoding="utf-8")),
                pd.read_csv(debug, low_memory=False),
                {
                    "quality_report_sha256": sha256_file(quality),
                    "association_debug_sha256": sha256_file(debug),
                    "prediction_xml_sha256": sha256_file(prediction),
                    "source_video_sha256": videos[video_key].video_sha256,
                    "gt_sha256": videos[video_key].gt_sha256,
                },
            )
        report, debug, hashes = cache[video_key]
        event["event_id"] = f"CURRENT_MAIN_E{index:03d}"
        event.update(_window_frame_evidence(report, event["start_frame"]))
        event.update(_window_debug_evidence(debug, event["start_frame"]))
        video = videos[video_key]
        event["gt_authority"] = video.gt_authority
        event["visibility_and_hidden_state"] = (
            "HIDDEN_OR_OCCLUSION_HOLD_AT_ONSET"
            if event["hidden_count_max"]
            else "VISIBLE_OR_NOT_EXPORTED"
        )
        event["detection_availability"] = (
            "PARTIAL_DROPOUT_AT_ONSET"
            if event["missing_detection_count_max"]
            else "FULL_COUNT_AT_ONSET"
        )
        event["motion_lk_availability"] = event.pop("lk_availability")
        event["track_birth_termination_state"] = (
            "BIRTH_BOUNDARY"
            if event["start_frame"] == 0
            else "TERMINAL_BOUNDARY"
            if event["end_frame"] >= video.frame_count - 1
            else "MID_VIDEO"
        )
        event["recovery_behavior"] = (
            "NO_RECOVERY_BEFORE_VIDEO_END"
            if event["end_frame"] >= video.frame_count - 1
            else "EVENT_ENDED_BEFORE_VIDEO_END"
        )
        event["terminal_swap"] = event["end_frame"] >= video.frame_count - 1
        event["permanent_swap"] = (
            event["terminal_swap"]
            or event["duration_frames"] >= PERMANENT_DURATION_FRAMES
        )
        event["causal_evidence_available_at_failure"] = (
            "PARTIAL_EXPORTED_CAUSAL_STATE"
            if event["association_rows_at_onset"]
            else "FRAME_SUMMARY_ONLY"
        )
        event["supporting_artifact_hashes"] = json.dumps(
            hashes,
            sort_keys=True,
            separators=(",", ":"),
        )
        primary, secondary = classify_event(event)
        if primary not in MECHANISMS:
            raise BaselineAuditError(f"undeclared mechanism: {primary}")
        event["primary_mechanism"] = primary
        event["secondary_mechanisms"] = secondary
        event["current_tracker_already_addresses"] = (
            "PARTIAL_OR_UNKNOWN_FROM_OBSERVATIONAL_BASELINE"
        )
        event["h1_targeted"] = H1_TARGETS.get(primary, "NO")
        event["h2_targeted"] = H2_TARGETS.get(primary, "NO")
        event["intervention_feasibility"] = (
            "LOW"
            if primary in {"GT_OR_EVALUATION_AMBIGUITY", "UNRESOLVED"}
            else "REQUIRES_FROZEN_DESIGN_AND_CONTROLS"
        )
        events.append(event)
    return events


def mechanism_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    authoritative_total = sum(
        int(event["wrong_id_matched_frames"])
        for event in events
        if event["gt_authority"] == "AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS"
    )
    for mechanism in MECHANISMS:
        subset = [event for event in events if event["primary_mechanism"] == mechanism]
        wrong = sum(int(event["wrong_id_matched_frames"]) for event in subset)
        authoritative = [
            event
            for event in subset
            if event["gt_authority"]
            == "AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS"
        ]
        causal_available = sum(
            event["causal_evidence_available_at_failure"]
            == "PARTIAL_EXPORTED_CAUSAL_STATE"
            for event in authoritative
        )
        rows.append(
            {
                "primary_mechanism": mechanism,
                "event_count": len(subset),
                "authoritative_event_count": len(authoritative),
                "wrong_id_frames": wrong,
                "authoritative_wrong_id_frames": sum(
                    int(event["wrong_id_matched_frames"])
                    for event in authoritative
                ),
                "total_duration": sum(
                    int(event["duration_frames"]) for event in subset
                ),
                "id_switches": sum(int(event["id_switch_rows"]) for event in subset),
                "permanent_swaps": sum(bool(event["permanent_swap"]) for event in subset),
                "terminal_swaps": sum(bool(event["terminal_swap"]) for event in subset),
                "source_videos_affected": len(
                    {event["video_key"] for event in subset}
                ),
                "causal_evidence_available_events": causal_available,
                "current_tracker_already_addresses": (
                    "PARTIAL_OR_UNKNOWN_FROM_OBSERVATIONAL_BASELINE"
                ),
                "h1_targeted": H1_TARGETS.get(mechanism, "NO"),
                "h2_targeted": H2_TARGETS.get(mechanism, "NO"),
                "intervention_feasibility": (
                    "LOW"
                    if mechanism in {"GT_OR_EVALUATION_AMBIGUITY", "UNRESOLVED"}
                    else "REQUIRES_FROZEN_DESIGN_AND_CONTROLS"
                ),
                "authoritative_wrong_id_percent": (
                    round(
                        100
                        * sum(
                            int(event["wrong_id_matched_frames"])
                            for event in authoritative
                        )
                        / authoritative_total,
                        6,
                    )
                    if authoritative_total
                    else 0.0
                ),
            }
        )
    return rows


def reconcile_historical(
    current_events: list[dict[str, Any]],
    historical_path: Path,
) -> list[dict[str, Any]]:
    historical = pd.read_csv(historical_path)
    rows = []
    for _, old in historical.iterrows():
        video = str(old["video_key"])
        if "_000216_" in video:
            status = "GT_INCONCLUSIVE"
            matches: list[dict[str, Any]] = []
        else:
            matches = [
                event
                for event in current_events
                if event["video_key"] == video
                and event["start_frame"] <= int(old["end_frame"]) + GAP_TOLERANCE_FRAMES
                and event["end_frame"] >= int(old["start_frame"]) - GAP_TOLERANCE_FRAMES
            ]
            if not matches:
                status = "RESOLVED_ON_CURRENT_MAIN"
            elif all(
                event["primary_mechanism"] != str(old["primary_mechanism"])
                for event in matches
            ):
                status = "CHANGED_MECHANISM"
            elif max(event["duration_frames"] for event in matches) < int(
                old["duration_frames"]
            ):
                status = "REDUCED_DURATION"
            else:
                status = "STILL_PRESENT"
        rows.append(
            {
                "historical_event_id": old["event_id"],
                "video_key": video,
                "historical_start_frame": int(old["start_frame"]),
                "historical_end_frame": int(old["end_frame"]),
                "historical_primary_mechanism": old["primary_mechanism"],
                "current_status": status,
                "current_event_ids": "|".join(
                    event["event_id"] for event in matches
                ),
                "historical_taxonomy_role": (
                    "HISTORICAL_MECHANISM_DISCOVERY_ONLY"
                ),
                "causal_attribution": (
                    "NOT_INFERRED_FROM_TEMPORAL_COMMIT_ORDER"
                ),
            }
        )
    return rows


def rank_mechanisms(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_wrong = max(
        (int(row["authoritative_wrong_id_frames"]) for row in summary),
        default=1,
    )
    max_events = max(
        (int(row["authoritative_event_count"]) for row in summary),
        default=1,
    )
    rows = []
    for row in summary:
        mechanism = str(row["primary_mechanism"])
        authoritative_events = int(row["authoritative_event_count"])
        wrong = int(row["authoritative_wrong_id_frames"])
        severity = 1.0 + 0.5 * int(row["permanent_swaps"]) + int(
            row["terminal_swaps"]
        )
        evidence = (
            int(row["causal_evidence_available_events"]) / authoritative_events
            if authoritative_events
            else 0.0
        )
        specificity = 0.2 if mechanism in {"UNRESOLVED", "OTHER_MEASURED"} else 0.8
        feasibility = (
            0.0
            if mechanism == "GT_OR_EVALUATION_AMBIGUITY"
            else 0.7
        )
        failed_family_penalty = (
            0.35
            if H1_TARGETS.get(mechanism) == "YES"
            or H2_TARGETS.get(mechanism) == "YES"
            else 1.0
        )
        score = (
            (wrong / max(max_wrong, 1))
            * (authoritative_events / max(max_events, 1))
            * severity
            * evidence
            * specificity
            * feasibility
            * failed_family_penalty
        )
        rows.append(
            {
                "primary_mechanism": mechanism,
                "authoritative_wrong_id_frames": wrong,
                "authoritative_event_count": authoritative_events,
                "permanent_terminal_severity_factor": round(severity, 6),
                "causal_evidence_factor": round(evidence, 6),
                "intervention_specificity_factor": specificity,
                "evaluation_feasibility_factor": feasibility,
                "failed_h1_h2_reuse_penalty": failed_family_penalty,
                "priority_score": round(score, 9),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["authoritative_wrong_id_frames"]),
            str(row["primary_mechanism"]),
        ),
    )


def tracking_decision(
    ranking: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> dict[str, Any]:
    top = ranking[0]
    detection_categories = {
        "DETECTION_MISS_OR_DROPOUT",
        "DETECTION_MERGE_OR_SPLIT",
    }
    authoritative_wrong = sum(
        int(row["authoritative_wrong_id_frames"]) for row in summary
    )
    detection_wrong = sum(
        int(row["authoritative_wrong_id_frames"])
        for row in summary
        if row["primary_mechanism"] in detection_categories
    )
    if not authoritative_wrong:
        outcome = "CURRENT_BASELINE_ALREADY_MEETS_QUALITY_TARGET"
    elif detection_wrong / authoritative_wrong >= 0.5:
        outcome = "IMPROVE_DETECTOR_FIRST"
    else:
        outcome = "NO_FURTHER_TRACKING_CHANGE_JUSTIFIED"
    return {
        "schema_version": "tracking.current_main_decision.v1",
        "decision": outcome,
        "next_hypothesis_name": None,
        "top_ranked_mechanism": top["primary_mechanism"],
        "causal_evidence_available_for_new_hypothesis": False,
        "reason": (
            "A fresh exact-main taxonomy was established, but this task does "
            "not establish every prerequisite for a new tracking hypothesis."
        ),
        "h1_family_status": "CLOSED_FOR_CURRENT_STUDY",
        "h2_status": "FAIL_NO_CURRENT_MAIN_STATE_LOSS",
        "new_implementation_authorized": False,
        "new_tracking_candidate_authorized": False,
        "validation_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
        "ready_for_next_hypothesis_design": False,
        "ready_for_tracking_implementation": False,
        "ready_to_promote": False,
    }


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
    historical_taxonomy: Path,
) -> None:
    from pig_behavior.evaluation.tracking.metrics import aggregate_metrics

    if output_root.exists():
        raise BaselineAuditError(f"refusing existing output root: {output_root}")
    producer_sha, tracking_tree = require_exact_tracking_authority()
    videos, lineage_sha = load_population(source_repo, lineage_manifest)
    cfg, detector_payload = detector_configuration(source_repo)
    config_payload = effective_config_payload()
    caches, cache_report = load_caches(videos, cache_root)
    output_root.mkdir(parents=True)
    metrics, identity_rows, mappings, replay_calls = run_predictions(
        source_repo,
        videos,
        caches,
        output_root,
    )
    aggregate = aggregate_metrics(metrics)
    per_video = []
    for metric in metrics:
        row = asdict(metric)
        row["wrong_id_matched_frames"] = max(
            0,
            int(metric.matches - metric.remapped_idtp),
        )
        per_video.append(row)
    aggregate_row = asdict(aggregate)
    aggregate_row["wrong_id_matched_frames"] = max(
        0,
        int(aggregate.matches - aggregate.remapped_idtp),
    )
    grouped = group_error_events(pd.DataFrame(identity_rows))
    events = enrich_events(
        grouped,
        output_root,
        {video.video_key: video for video in videos},
    )
    wrong_rows = sum(
        "mismatch" in str(row["event"]) for row in identity_rows
    )
    switch_rows = sum(
        "id_switch" in str(row["event"]) for row in identity_rows
    )
    if sum(int(event["wrong_id_matched_frames"]) for event in events) != wrong_rows:
        raise BaselineAuditError("wrong-ID event conservation failed")
    if sum(int(event["id_switch_rows"]) for event in events) != switch_rows:
        raise BaselineAuditError("ID-switch event conservation failed")
    summary = mechanism_summary(events)
    reconciliation = reconcile_historical(events, historical_taxonomy)
    ranking = rank_mechanisms(summary)
    decision = tracking_decision(ranking, summary)
    mp4_count = len(list(output_root.rglob("*.mp4")))
    if mp4_count:
        raise BaselineAuditError("baseline run produced MP4")
    write_csv(
        output_root / "CURRENT_MAIN_PER_VIDEO_METRICS.csv",
        per_video,
        list(per_video[0]),
    )
    identity_fields = list(identity_rows[0]) if identity_rows else [
        "video_stem",
        "frame",
        "gt_id",
        "pred_id",
        "event",
    ]
    write_csv(
        output_root / "CURRENT_MAIN_IDENTITY_ERROR_ROWS.csv",
        identity_rows,
        identity_fields,
    )
    event_fields = sorted({key for event in events for key in event})
    write_csv(
        output_root / "CURRENT_MAIN_IDENTITY_ERROR_EVENTS.csv",
        events,
        event_fields,
    )
    write_json(
        output_root / "CURRENT_MAIN_ERROR_EVENT_INTEGRITY.json",
        {
            "schema_version": "tracking.current_main_event_integrity.v1",
            "result": "PASS",
            "identity_error_rows": len(identity_rows),
            "wrong_id_rows": wrong_rows,
            "id_switch_rows": switch_rows,
            "connected_events": len(events),
            "event_wrong_id_rows": sum(
                int(event["wrong_id_matched_frames"]) for event in events
            ),
            "event_id_switch_rows": sum(
                int(event["id_switch_rows"]) for event in events
            ),
            "rows_lost": 0,
            "rows_double_counted": 0,
        },
    )
    write_csv(
        output_root / "CURRENT_MAIN_PRIMARY_FAILURE_MECHANISM_SUMMARY.csv",
        summary,
        list(summary[0]),
    )
    swap_rows = [
        {
            "event_id": event["event_id"],
            "video_key": event["video_key"],
            "primary_mechanism": event["primary_mechanism"],
            "duration_frames": event["duration_frames"],
            "permanent_swap": event["permanent_swap"],
            "terminal_swap": event["terminal_swap"],
            "gt_authority": event["gt_authority"],
        }
        for event in events
        if event["permanent_swap"] or event["terminal_swap"]
    ]
    write_csv(
        output_root / "CURRENT_MAIN_PERMANENT_TERMINAL_SWAP_SUMMARY.csv",
        swap_rows,
        [
            "event_id",
            "video_key",
            "primary_mechanism",
            "duration_frames",
            "permanent_swap",
            "terminal_swap",
            "gt_authority",
        ],
    )
    write_csv(
        output_root / "HISTORICAL_TO_CURRENT_EVENT_RECONCILIATION.csv",
        reconciliation,
        list(reconciliation[0]),
    )
    write_csv(
        output_root / "CURRENT_MAIN_NEXT_HYPOTHESIS_RANKING.csv",
        ranking,
        list(ranking[0]),
    )
    write_json(
        output_root / "CURRENT_MAIN_TRACKING_DECISION.json",
        decision,
    )
    metrics_payload = {
        "schema_version": "tracking.current_main_baseline_metrics.v1",
        "metric_contract": {
            "profile": PROFILE,
            "include_hidden": True,
            "iou_threshold": IOU_THRESHOLD,
            "gap_tolerance_frames": GAP_TOLERANCE_FRAMES,
            "identity_reporting": "GLOBAL_REMAP_PER_VIDEO_THEN_AGGREGATE",
        },
        "aggregate": aggregate_row,
        "identity_mappings": mappings,
    }
    write_json(
        output_root / "CURRENT_MAIN_BASELINE_METRICS.json",
        metrics_payload,
    )
    manifest = {
        "schema_version": "tracking.current_main_baseline_run.v1",
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": producer_sha,
        "tracking_tree_object": tracking_tree,
        "source_lineage_sha256": lineage_sha,
        "profile": PROFILE,
        "effective_config_sha256": canonical_hash(config_payload),
        "effective_config": config_payload,
        "detector_semantic_config_sha256": canonical_hash(detector_payload),
        "detector_semantic_config": detector_payload,
        "detector_weight_sha256": sha256_file(Path(cfg.weights_path)),
        "cache_authority": cache_report,
        "baseline_videos_total": 13,
        "baseline_videos_completed": len(metrics),
        "include_hidden": True,
        "causal_timing_policy": "causal_framewise",
        "output_delay_frames": 0,
        "future_frames_used": False,
        "offline_repair": False,
        "post_video_smoothing": False,
        "detector_inference_calls_during_tracking": 0,
        "cache_replay_calls_during_tracking": replay_calls,
        "h1_h2_validation_execution": False,
        "h1_h2_validation_roles_consumed": False,
        "run_root_mp4_count": mp4_count,
        "repeatability": "NOT_RUN_NOT_REQUIRED_BY_FROZEN_POLICY",
    }
    write_json(
        output_root / "CURRENT_MAIN_BASELINE_RUN_MANIFEST.json",
        manifest,
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
            "schema_version": "tracking.current_main_baseline_inventory.v1",
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
    parser.add_argument(
        "--historical-taxonomy",
        type=Path,
        default=HISTORICAL_TAXONOMY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute(
        args.source_repo.resolve(),
        args.lineage_manifest.resolve(),
        args.cache_root.resolve(),
        args.output_root.resolve(),
        args.historical_taxonomy.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
