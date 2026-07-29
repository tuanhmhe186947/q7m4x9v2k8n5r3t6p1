#!/usr/bin/env python3
"""Execute and evaluate the frozen four-method reconciliation development set.

This tool starts rf_hybrid at the frozen realtime_fast tracklet boundary. It
does not invoke a detector or tracker, and it treats the surviving historical
XMLs as the bytetrack_raw and hybrid_bytetrack prediction authorities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.tracking import (  # noqa: E402
    evaluate_b0_b1_r0_standard_v2 as standard_v2,
)
from scripts.tracking import (  # noqa: E402
    generate_b0_b1_frozen_predictions as prediction_tools,
)
from scripts.tracking import (  # noqa: E402
    generate_r1_rf_hybrid_predictions as r1_tools,
)

from pig_behavior.evaluation.tracking.cvat_io import (  # noqa: E402
    parse_cvat_video_xml,
)
from pig_behavior.tracking.exporters.cvat_xml import (  # noqa: E402
    write_cvat_video_xml,
)
from pig_behavior.tracking.masks import load_mask  # noqa: E402
from pig_behavior.tracking.refinement import (  # noqa: E402
    _shape_attributes_dict,
)
from pig_behavior.tracking.rf_hybrid_transfer import (  # noqa: E402
    RF_HYBRID_TRANSFER_STAGE_IDS,
    apply_rf_hybrid_transfer,
    build_rf_hybrid_transfer_config,
    canonical_transfer_hash,
    rf_hybrid_transfer_config_hash,
    write_rf_hybrid_artifacts,
)

DATE = "20260729"
EXPECTED_CODE_BASE = "a4bfc0ca82c0e9a8038bc47860444c17a97ab658"
EXPECTED_TRANSFER_CONFIG_HASH = (
    "d583450c5b462ffdb65835d4aa2e9352c88afb96d29176ab2866e6f1cdb0ad23"
)
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
SELECTED_SKILLS = (
    "tracking-experiment-guardian",
    "experiment-lineage-reproducibility",
    "safe-refactor-test-guardian",
)

B0_RELATIVE = Path(
    "outputs/tracking/three_mode_historical_reconstruction_20260729/"
    "frozen_predictions/B0_historical"
)
HYBRID_RELATIVE = Path(
    "outputs/tracking/historical_h5b_h4_frozen_predictions_20260728/"
    "predictions"
)
R0_RELATIVE = Path(
    "outputs/tracking/three_mode_historical_reconstruction_20260729/"
    "frozen_predictions/R0_historical_RF_ACC23"
)
RAW_TRACKLET_RELATIVE = Path(
    "outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/"
    "R1_rf_hybrid_offline/raw_core_snapshots"
)
MASK_RELATIVE = Path("data/annotations/scene/mask.png")
WEIGHTS_RELATIVE = Path("models/detector/pig_detector_yolov8.pt")

AUTHORITY_RELATIVES = (
    Path(
        "docs/tracking/reconciliation/"
        "STATE_7_RF_HYBRID_TRANSFER_AUTHORITY_20260729.json"
    ),
    Path(
        "docs/tracking/reconciliation/"
        "STATE_6_RF_HYBRID_PORTABILITY_AUTHORITY_20260729.json"
    ),
    Path(
        "docs/tracking/historical_hybrid_best_recovery/"
        "HISTORICAL_HYBRID_BEST_LINEAGE_RECOVERY_AUTHORITY_20260729.json"
    ),
    Path(
        "docs/tracking/three_mode_historical_reconstruction/"
        "B0_HISTORICAL_RECONSTRUCTION_AUTHORITY_20260729.json"
    ),
    Path(
        "docs/tracking/three_mode_historical_reconstruction/"
        "R0_HISTORICAL_RECONSTRUCTION_AUTHORITY_20260729.json"
    ),
)

TRANSFER_DECISION_RULE = {
    "TRANSFER_SIGNAL_POSITIVE": (
        "HOTA and IDF1 do not decrease; IDSW_STANDARD and wrong-identity "
        "frames do not increase; at least one is strictly better; and no "
        "changed row is harmful."
    ),
    "NO_TRANSFER_SIGNAL": (
        "The four primary quantities are unchanged and no final row changes."
    ),
    "TRANSFER_DEGRADES_RF": (
        "HOTA and IDF1 both decrease while IDSW_STANDARD and wrong-identity "
        "frames both do not improve."
    ),
    "TRANSFER_SIGNAL_MIXED": (
        "Every other contract-valid result, including opposing quality and "
        "identity-severity movements."
    ),
}


class State8Error(RuntimeError):
    """Fail-closed State 8 authority error."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise State8Error(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic row table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def package_version(name: str) -> str:
    """Return a package version without importing optional heavy packages."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def clean_commit_guard(worktree_repo: Path) -> str:
    """Require a clean worktree descended from the frozen State 7 commit."""

    code_sha = git_output(worktree_repo, "rev-parse", "HEAD")
    dirty = git_output(worktree_repo, "status", "--porcelain")
    if dirty:
        raise State8Error("State 8 execution requires a clean worktree")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_CODE_BASE, code_sha],
        cwd=worktree_repo,
        check=False,
    )
    if ancestor.returncode != 0:
        raise State8Error("State 8 code is not descended from State 7")
    return code_sha


def input_record(path: Path, role: str) -> dict[str, Any]:
    """Hash one immutable State 8 input."""

    if not path.is_file():
        raise State8Error(f"Missing {role}: {path}")
    return {
        "path": str(path),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _shape_key(shape: dict[str, Any]) -> tuple[int, str]:
    return int(shape["frame"]), str(shape["label"])


def _public_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in shape.items()
        if not str(key).startswith("_")
    }


def _identity(shape: dict[str, Any] | None) -> str:
    if shape is None:
        return ""
    return str(_shape_attributes_dict(shape).get("ID", ""))


def _bbox_iou(left: list[float], right: tuple[float, ...]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(
        0.0,
        left[3] - left[1],
    )
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0,
        right[3] - right[1],
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def contiguous_episode_count(
    keys: set[tuple[str, int, str]],
) -> int:
    """Count final contiguous changed-frame groups per video and track label."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for video, frame, label in keys:
        grouped[(video, label)].append(frame)
    count = 0
    for frames in grouped.values():
        previous: int | None = None
        for frame in sorted(set(frames)):
            if previous is None or frame != previous + 1:
                count += 1
            previous = frame
    return count


def classify_transfer_signal(
    realtime: dict[str, float],
    candidate: dict[str, float],
    *,
    harmful_changes: int,
    final_changed_rows: int,
) -> str:
    """Apply the predeclared State 8 development decision rule."""

    hota_delta = candidate["hota"] - realtime["hota"]
    idf1_delta = candidate["idf1"] - realtime["idf1"]
    idsw_delta = (
        candidate["idsw_standard"] - realtime["idsw_standard"]
    )
    wrong_delta = (
        candidate["wrong_id_matched_frames"]
        - realtime["wrong_id_matched_frames"]
    )
    if (
        hota_delta >= 0.0
        and idf1_delta >= 0.0
        and idsw_delta <= 0.0
        and wrong_delta <= 0.0
        and any(
            value != 0.0
            for value in (hota_delta, idf1_delta, idsw_delta, wrong_delta)
        )
        and harmful_changes == 0
    ):
        return "TRANSFER_SIGNAL_POSITIVE"
    if (
        hota_delta == 0.0
        and idf1_delta == 0.0
        and idsw_delta == 0.0
        and wrong_delta == 0.0
        and final_changed_rows == 0
    ):
        return "NO_TRANSFER_SIGNAL"
    if (
        hota_delta < 0.0
        and idf1_delta < 0.0
        and idsw_delta >= 0.0
        and wrong_delta >= 0.0
    ):
        return "TRANSFER_DEGRADES_RF"
    return "TRANSFER_SIGNAL_MIXED"


def _evaluation_status(
    evaluation: Any,
) -> tuple[
    dict[tuple[int, str], str],
    dict[tuple[int, str], str],
]:
    statuses: dict[tuple[int, str], str] = {}
    gt_matches: dict[tuple[int, str], str] = {}
    result = evaluation.episode_result
    for row in result.authoritative_correct_rows:
        statuses[(row.frame, row.pred_id)] = "correct"
        gt_matches[(row.frame, row.pred_id)] = row.gt_id
    for row in result.authoritative_wrong_rows:
        statuses[(row.frame, row.pred_id)] = "wrong"
        gt_matches[(row.frame, row.pred_id)] = row.gt_id
    for ambiguity in result.ambiguous_rows:
        row = ambiguity.row
        statuses[(row.frame, row.pred_id)] = "ambiguous"
        gt_matches[(row.frame, row.pred_id)] = row.gt_id
    return statuses, gt_matches


def _gt_bbox_lookup(gt_path: Path) -> dict[tuple[int, str], tuple[float, ...]]:
    parsed = parse_cvat_video_xml(
        gt_path,
        include_hidden=True,
        start_frame=0,
        end_frame=EXPECTED_FRAMES - 1,
    )
    return {
        (frame, obj.obj_id): tuple(float(value) for value in obj.bbox)
        for frame, objects in parsed.items()
        for obj in objects
    }


def classify_final_changes(
    generated: list[dict[str, Any]],
    evaluations: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Classify final row changes using the frozen Standard V2 matches."""

    raw_evaluations = {
        item.metrics.video_stem: item
        for item in evaluations["realtime_fast"]
    }
    hybrid_evaluations = {
        item.metrics.video_stem: item
        for item in evaluations["rf_hybrid"]
    }
    rows: list[dict[str, Any]] = []
    counts = {"beneficial": 0, "harmful": 0, "neutral": 0}
    for video in generated:
        video_key = str(video["video_key"])
        raw_status, raw_gt = _evaluation_status(raw_evaluations[video_key])
        new_status, new_gt = _evaluation_status(
            hybrid_evaluations[video_key]
        )
        gt_bboxes = _gt_bbox_lookup(Path(video["gt_path"]))
        for change in video["final_changes"]:
            frame = int(change["frame"])
            old_id = str(change["old_identity"])
            new_id = str(change["new_identity"])
            old_state = raw_status.get((frame, old_id), "unmatched")
            new_state = new_status.get((frame, new_id), "unmatched")
            category = "neutral"
            evidence = "IDENTITY_STATUS_UNCHANGED_OR_NONAUTHORITATIVE"
            if old_state != "correct" and new_state == "correct":
                category = "beneficial"
                evidence = "BECAME_AUTHORITATIVE_CORRECT"
            elif old_state == "correct" and new_state != "correct":
                category = "harmful"
                evidence = "LOST_AUTHORITATIVE_CORRECT_MATCH"
            elif (
                old_state == "correct"
                and new_state == "correct"
                and change["old_bbox"] is not None
                and change["new_bbox"] is not None
            ):
                old_gt_id = raw_gt[(frame, old_id)]
                new_gt_id = new_gt[(frame, new_id)]
                if old_gt_id == new_gt_id:
                    gt_bbox = gt_bboxes[(frame, old_gt_id)]
                    old_iou = _bbox_iou(change["old_bbox"], gt_bbox)
                    new_iou = _bbox_iou(change["new_bbox"], gt_bbox)
                    if new_iou > old_iou:
                        category = "beneficial"
                        evidence = "CORRECT_MATCH_IOU_INCREASED"
                    elif new_iou < old_iou:
                        category = "harmful"
                        evidence = "CORRECT_MATCH_IOU_DECREASED"
            counts[category] += 1
            rows.append(
                {
                    **change,
                    "old_match_status": old_state,
                    "new_match_status": new_state,
                    "classification": category,
                    "classification_evidence": evidence,
                }
            )
    return rows, counts


def _aggregate_metric_row(
    dataframe: Any,
    method_id: str,
) -> dict[str, float]:
    match = dataframe.loc[dataframe["arm"] == method_id]
    if len(match) != 1:
        raise State8Error(f"Missing aggregate metrics for {method_id}")
    row = match.iloc[0]
    return {
        key: float(row[key])
        for key in (
            "hota",
            "deta",
            "assa",
            "loca",
            "idf1",
            "idp",
            "idr",
            "idsw_standard",
            "fp",
            "fn",
            "fragments",
            "wrong_id_matched_frames",
            "wrong_id_matched_seconds",
            "recovered_identity_error_episode_count",
            "terminal_identity_error_episode_count",
            "persistent_pairwise_identity_swap_count",
        )
    }


def _prediction_paths(
    source_repo: Path,
    output_root: Path,
    video_key: str,
) -> dict[str, str]:
    return {
        "bytetrack_raw": str(source_repo / B0_RELATIVE / f"{video_key}.xml"),
        "hybrid_bytetrack": str(
            source_repo / HYBRID_RELATIVE / f"{video_key}.xml"
        ),
        "realtime_fast": str(
            output_root / "predictions/realtime_fast" / f"{video_key}.xml"
        ),
        "rf_hybrid": str(
            output_root / "predictions/rf_hybrid" / f"{video_key}.xml"
        ),
    }


def _arms(source_repo: Path, output_root: Path) -> tuple[Any, ...]:
    unknown_hash = "BOUND_BY_PER_FILE_PREDICTION_MANIFEST"
    detector_hash = sha256_file(source_repo / WEIGHTS_RELATIVE)
    return (
        standard_v2.ArmSpec(
            arm="bytetrack_raw",
            profile="bytetrack_raw",
            prediction_root=source_repo / B0_RELATIVE,
            authority_path=source_repo / AUTHORITY_RELATIVES[3],
            artifact_sha256=unknown_hash,
            config_sha256="HISTORICAL_ORIGINAL_BASELINE_AUTHORITY",
            detector_cadence="EVERY_FRAME_LIVE_YOLO_TRACK",
            detector_authority_sha256=detector_hash,
        ),
        standard_v2.ArmSpec(
            arm="hybrid_bytetrack",
            profile="hybrid_bytetrack",
            prediction_root=source_repo / HYBRID_RELATIVE,
            authority_path=source_repo / AUTHORITY_RELATIVES[2],
            artifact_sha256=unknown_hash,
            config_sha256="FULL_ACCEPTED_HISTORICAL_LINEAGE",
            detector_cadence="EVERY_FRAME_LIVE_YOLO_TRACK",
            detector_authority_sha256=detector_hash,
        ),
        standard_v2.ArmSpec(
            arm="realtime_fast",
            profile="realtime_fast",
            prediction_root=output_root / "predictions/realtime_fast",
            authority_path=source_repo / AUTHORITY_RELATIVES[4],
            artifact_sha256=unknown_hash,
            config_sha256="FROZEN_RF_ACC23_CORE",
            detector_cadence="EVERY_SECOND_FRAME",
            detector_authority_sha256=detector_hash,
        ),
        standard_v2.ArmSpec(
            arm="rf_hybrid",
            profile="rf_hybrid",
            prediction_root=output_root / "predictions/rf_hybrid",
            authority_path=source_repo / AUTHORITY_RELATIVES[0],
            artifact_sha256=unknown_hash,
            config_sha256=EXPECTED_TRANSFER_CONFIG_HASH,
            detector_cadence="SHARED_REALTIME_FAST_EVIDENCE",
            detector_authority_sha256=detector_hash,
        ),
    )


def _final_changes(
    video_key: str,
    raw_shapes: list[dict[str, Any]],
    final_shapes: list[dict[str, Any]],
    declared_keys: set[tuple[int, str]],
) -> tuple[list[dict[str, Any]], set[tuple[str, int, str]]]:
    raw = {_shape_key(shape): shape for shape in raw_shapes}
    final = {_shape_key(shape): shape for shape in final_shapes}
    changed_keys = {
        key
        for key in set(raw) | set(final)
        if _public_shape(raw.get(key, {}))
        != _public_shape(final.get(key, {}))
    }
    outside = changed_keys - declared_keys
    if outside:
        raise State8Error(
            f"RF changes outside declared ledger scope: {video_key}"
        )
    rows = []
    for frame, label in sorted(changed_keys):
        old_shape = raw.get((frame, label))
        new_shape = final.get((frame, label))
        rows.append(
            {
                "video": video_key,
                "frame": frame,
                "label": label,
                "old_identity": _identity(old_shape),
                "new_identity": _identity(new_shape),
                "old_bbox": (
                    [float(value) for value in old_shape["points"]]
                    if old_shape is not None
                    else None
                ),
                "new_bbox": (
                    [float(value) for value in new_shape["points"]]
                    if new_shape is not None
                    else None
                ),
            }
        )
    global_keys = {
        (video_key, frame, label) for frame, label in changed_keys
    }
    return rows, global_keys


def _recursive_inventory(root: Path) -> list[dict[str, Any]]:
    excluded = {
        root / "artifact_manifest.json",
        root / "STATE_8_DEVELOPMENT_EVALUATION_DECISION.json",
    }
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in excluded
    ]


def _prepare_manifest(
    source_repo: Path,
    worktree_repo: Path,
    output_root: Path,
    code_sha: str,
    videos: list[Any],
) -> list[dict[str, Any]]:
    if output_root.exists():
        raise State8Error(f"Refusing existing output root: {output_root}")
    output_root.mkdir(parents=True)
    authority_inputs = [
        input_record(source_repo / path, "SCIENTIFIC_AUTHORITY")
        for path in AUTHORITY_RELATIVES
    ]
    shared_inputs = [
        input_record(source_repo / MASK_RELATIVE, "SCENE_MASK"),
        input_record(source_repo / WEIGHTS_RELATIVE, "DETECTOR_CONTRACT"),
    ]
    video_inputs = []
    for video in videos:
        video_inputs.extend(
            [
                input_record(video.video_path, "DEVELOPMENT_VIDEO"),
                input_record(video.gt_path, "DEVELOPMENT_GT"),
                input_record(
                    source_repo
                    / RAW_TRACKLET_RELATIVE
                    / f"{video.video_key}.rf_raw_track_output.json",
                    "FROZEN_REALTIME_FAST_TRACKLETS",
                ),
                input_record(
                    source_repo / B0_RELATIVE / f"{video.video_key}.xml",
                    "BYTETRACK_RAW_PREDICTION_AUTHORITY",
                ),
                input_record(
                    source_repo / HYBRID_RELATIVE / f"{video.video_key}.xml",
                    "HYBRID_BYTETRACK_PREDICTION_AUTHORITY",
                ),
                input_record(
                    source_repo / R0_RELATIVE / f"{video.video_key}.xml",
                    "REALTIME_FAST_PREDICTION_AUTHORITY",
                ),
            ]
        )
    input_inventory = authority_inputs + shared_inputs + video_inputs
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "tracking.reconciliation.state8_run.v1",
            "state": "STATE_8_DEVELOPMENT_EVALUATION",
            "status": "PLANNED",
            "date": DATE,
            "code_sha": code_sha,
            "state7_base_sha": EXPECTED_CODE_BASE,
            "source_repo": str(source_repo),
            "worktree_repo": str(worktree_repo),
            "output_root": str(output_root),
            "selected_skills": list(SELECTED_SKILLS),
            "method_ids": [
                "bytetrack_raw",
                "hybrid_bytetrack",
                "realtime_fast",
                "rf_hybrid",
            ],
            "comparisons": {
                "hybrid_bytetrack_vs_bytetrack_raw": (
                    "FULL_ACCEPTED_HISTORICAL_OPTIMIZATION_GAIN"
                ),
                "realtime_fast_vs_bytetrack_raw": (
                    "COMPLETE_CAUSAL_METHOD_COMPARISON"
                ),
                "rf_hybrid_vs_realtime_fast": (
                    "HYBRID_MECHANISM_TRANSFER_EFFECT"
                ),
            },
            "transfer_config_hash": rf_hybrid_transfer_config_hash(),
            "transfer_stage_ids": list(RF_HYBRID_TRANSFER_STAGE_IDS),
            "decision_rule": TRANSFER_DECISION_RULE,
            "evaluation_contract": "TRACKING_EVALUATOR_STANDARD_V2",
            "identity_contract": "IDENTITY_ERROR_EPISODES_V2",
            "include_hidden": True,
            "detection_iou_threshold": 0.5,
            "config_variants_executed": 1,
            "package_variants_executed": 1,
            "post_result_parameter_tuning": 0,
            "per_video_overrides": 0,
            "current_b1_used": False,
            "unseen_files_accessed": 0,
            "detector_executions": 0,
            "tracker_executions": 0,
            "input_inventory_sha256": canonical_transfer_hash(
                input_inventory
            ),
        },
    )
    write_json(
        output_root / "input_artifact_manifest.json",
        {
            "schema_version": "tracking.reconciliation.state8_inputs.v1",
            "artifact_count": len(input_inventory),
            "artifacts": input_inventory,
            "canonical_sha256": canonical_transfer_hash(input_inventory),
        },
    )
    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy",
                    "opencv-python",
                    "pandas",
                    "scipy",
                )
            },
            "device": "CPU_POST_TRACKLET_TRANSFER_AND_EVALUATION_ONLY",
            "detector_runtime": "NOT_EXECUTED",
            "tracker_runtime": "NOT_EXECUTED",
        },
    )
    return input_inventory


def execute(
    source_repo: Path,
    worktree_repo: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the one frozen State 8 transfer and Standard V2 evaluation."""

    code_sha = clean_commit_guard(worktree_repo)
    config_hash = rf_hybrid_transfer_config_hash()
    if config_hash != EXPECTED_TRANSFER_CONFIG_HASH:
        raise State8Error("RF hybrid transfer config hash changed")
    videos, _manifest = r1_tools.locked_population(source_repo)
    if len(videos) != EXPECTED_VIDEOS:
        raise State8Error("Development population is not 13 videos")
    input_inventory = _prepare_manifest(
        source_repo,
        worktree_repo,
        output_root,
        code_sha,
        videos,
    )
    for relative in (
        "predictions/realtime_fast",
        "predictions/rf_hybrid",
        "per_video",
    ):
        (output_root / relative).mkdir(parents=True)

    transfer_cfg = build_rf_hybrid_transfer_config()
    mask_path = source_repo / MASK_RELATIVE
    generated: list[dict[str, Any]] = []
    all_changed_keys: set[tuple[str, int, str]] = set()
    aggregate_ledger: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    transfer_started = time.perf_counter()
    for video in videos:
        raw_path = (
            source_repo
            / RAW_TRACKLET_RELATIVE
            / f"{video.video_key}.rf_raw_track_output.json"
        )
        raw_payload = load_json(raw_path)
        if raw_payload.get("source_core") != "realtime_fast":
            raise State8Error("Raw tracklets do not name realtime_fast")
        if raw_payload.get("video_key") != video.video_key:
            raise State8Error("Raw tracklet video key mismatch")
        raw_shapes = list(raw_payload["shapes"])
        raw_hash_before = canonical_transfer_hash(raw_shapes)
        mask = load_mask(
            mask_path,
            video.width,
            video.height,
            transfer_cfg,
        )
        result = apply_rf_hybrid_transfer(
            raw_shapes,
            video.width,
            video.height,
            mask,
            video=video.video_key,
        )
        if canonical_transfer_hash(raw_shapes) != raw_hash_before:
            raise State8Error("Frozen realtime_fast tracklets were mutated")
        if result.transfer_config_hash != config_hash:
            raise State8Error("Per-video transfer config drift")
        video_root = output_root / "per_video" / video.video_key
        video_root.mkdir(parents=True)
        realtime_json = video_root / "realtime_fast_output.json"
        hybrid_json = video_root / "rf_hybrid_output.json"
        ledger_json = video_root / "rf_hybrid_change_ledger.json"
        write_rf_hybrid_artifacts(
            realtime_fast_path=realtime_json,
            rf_hybrid_path=hybrid_json,
            ledger_path=ledger_json,
            video=video.video_key,
            raw_shapes=raw_shapes,
            result=result,
        )
        realtime_xml = (
            output_root
            / "predictions/realtime_fast"
            / f"{video.video_key}.xml"
        )
        hybrid_xml = (
            output_root
            / "predictions/rf_hybrid"
            / f"{video.video_key}.xml"
        )
        write_cvat_video_xml(
            realtime_xml,
            raw_shapes,
            video.video_path,
            video.width,
            video.height,
            EXPECTED_FRAMES,
        )
        write_cvat_video_xml(
            hybrid_xml,
            result.shapes,
            video.video_path,
            video.width,
            video.height,
            EXPECTED_FRAMES,
        )
        frozen_r0 = source_repo / R0_RELATIVE / f"{video.video_key}.xml"
        generated_r0_record = prediction_tools.xml_structural_record(
            realtime_xml,
            video_key=video.video_key,
            width=video.width,
            height=video.height,
        )
        frozen_r0_record = prediction_tools.xml_structural_record(
            frozen_r0,
            video_key=video.video_key,
            width=video.width,
            height=video.height,
        )
        if (
            generated_r0_record["canonical_row_sha256"]
            != frozen_r0_record["canonical_row_sha256"]
        ):
            raise State8Error(
                f"Realtime snapshot parity failed: {video.video_key}"
            )
        ledger = load_json(ledger_json)
        declared_keys = {
            (int(row["frame"]), str(row["label"]))
            for event in ledger["changes"]
            for field in ("old_bbox", "new_bbox")
            for row in event[field]
        }
        final_changes, changed_keys = _final_changes(
            video.video_key,
            raw_shapes,
            result.shapes,
            declared_keys,
        )
        all_changed_keys.update(changed_keys)
        aggregate_ledger.extend(ledger["changes"])
        stage_rows.extend(
            {
                "video": video.video_key,
                **row,
            }
            for row in ledger["stage_activation"]
        )
        generated.append(
            {
                "video_key": video.video_key,
                "gt_path": str(video.gt_path),
                "source_video_path": str(video.video_path),
                "source_video_sha256": video.video_sha256,
                "gt_sha256": video.gt_sha256,
                "frame_count": EXPECTED_FRAMES,
                "frames_per_second": 30.0,
                "mechanism_ranking_eligibility": (
                    video.mechanism_ranking_eligibility
                ),
                "input_authority_hash": result.input_authority_hash,
                "output_authority_hash": result.output_authority_hash,
                "realtime_fast_xml_sha256": sha256_file(realtime_xml),
                "rf_hybrid_xml_sha256": sha256_file(hybrid_xml),
                "frozen_r0_canonical_row_sha256": frozen_r0_record[
                    "canonical_row_sha256"
                ],
                "realtime_fast_canonical_row_sha256": generated_r0_record[
                    "canonical_row_sha256"
                ],
                "r0_snapshot_parity": "EXACT_SEMANTIC",
                "final_changes": final_changes,
                "final_changed_row_count": len(final_changes),
                "stage_change_episode_count": len(ledger["changes"]),
                "prediction_paths": _prediction_paths(
                    source_repo,
                    output_root,
                    video.video_key,
                ),
            }
        )
        print(
            f"STATE8_TRANSFER {video.video_key} "
            f"changed_rows={len(final_changes)}",
            flush=True,
        )
    transfer_seconds = time.perf_counter() - transfer_started

    video_rows = [
        {
            "video_key": row["video_key"],
            "source_video_path": row["source_video_path"],
            "source_video_sha256": row["source_video_sha256"],
            "gt_path": row["gt_path"],
            "gt_sha256": row["gt_sha256"],
            "frame_start": 0,
            "frame_end": EXPECTED_FRAMES - 1,
            "frame_count": EXPECTED_FRAMES,
            "frames_per_second": row["frames_per_second"],
            "visible_gt_rows": None,
            "hidden_gt_rows": None,
            "sequence_boundary": row["video_key"],
            "aggregate_inclusion_status": "DEVELOPMENT_AUTHORITY",
            "gt_authority_status": "FROZEN_DEVELOPMENT_GT",
            "mechanism_ranking_eligibility": row[
                "mechanism_ranking_eligibility"
            ],
            "prediction_paths": row["prediction_paths"],
        }
        for row in generated
    ]
    arms = _arms(source_repo, output_root)
    evaluator_code_sha = git_output(worktree_repo, "rev-parse", "HEAD")
    evaluation_started = time.perf_counter()
    pass1 = standard_v2.evaluate_pass(
        output_root / "standard_v2/pass_1",
        arms,
        video_rows,
        evaluator_code_sha=evaluator_code_sha,
        reverse_inputs=False,
    )
    pass2 = standard_v2.evaluate_pass(
        output_root / "standard_v2/pass_2",
        arms,
        video_rows,
        evaluator_code_sha=evaluator_code_sha,
        reverse_inputs=True,
    )
    determinism = standard_v2.compare_passes(pass1, pass2)
    evaluation_seconds = time.perf_counter() - evaluation_started
    if determinism["reevaluation_repeatability"] != "PASS":
        raise State8Error("Standard V2 two-pass determinism failed")
    conservation = pass1["conservation"]
    if any(
        conservation[key] != "PASS"
        for key in (
            "wrong_id_row_conservation",
            "tp_fp_fn_conservation",
            "multi_video_boundary_status",
        )
    ):
        raise State8Error("Standard V2 conservation failed")

    classified_rows, classified_counts = classify_final_changes(
        generated,
        pass1["evaluations"],
    )
    write_csv(
        output_root / "rf_hybrid_final_change_classification.csv",
        classified_rows,
    )
    write_json(
        output_root / "rf_hybrid_change_ledger.json",
        {
            "schema_version": "tracking.rf_hybrid_change_ledger.aggregate.v1",
            "method_id": "rf_hybrid",
            "source_method_id": "realtime_fast",
            "transfer_config_hash": config_hash,
            "stage_change_episode_count": len(aggregate_ledger),
            "changes": aggregate_ledger,
        },
    )
    write_csv(output_root / "rf_hybrid_stage_activation.csv", stage_rows)

    aggregate = pass1["aggregate_dataframe"]
    metrics = {
        method_id: _aggregate_metric_row(aggregate, method_id)
        for method_id in (
            "bytetrack_raw",
            "hybrid_bytetrack",
            "realtime_fast",
            "rf_hybrid",
        )
    }
    decision = classify_transfer_signal(
        metrics["realtime_fast"],
        metrics["rf_hybrid"],
        harmful_changes=classified_counts["harmful"],
        final_changed_rows=len(all_changed_keys),
    )
    changed_tracks = {
        (row["video"], row["label"]) for row in classified_rows
    }
    change_summary = {
        "schema_version": "tracking.reconciliation.rf_hybrid_changes.v1",
        "frames_modified": len(
            {(video, frame) for video, frame, _label in all_changed_keys}
        ),
        "tracks_modified": len(changed_tracks),
        "episodes_opened": len(aggregate_ledger),
        "episodes_changed": contiguous_episode_count(all_changed_keys),
        "beneficial_changes": classified_counts["beneficial"],
        "harmful_changes": classified_counts["harmful"],
        "neutral_changes": classified_counts["neutral"],
        "changes_outside_declared_episode_scope": 0,
        "final_changed_rows": len(all_changed_keys),
        "classification_unit": "FINAL_CHANGED_VIDEO_FRAME_TRACK_ROW",
        "classification_rule": (
            "Authoritative correct-match transitions first; for rows correct "
            "in both arms, bbox IoU movement to the same GT determines sign."
        ),
    }
    write_json(output_root / "RF_HYBRID_CHANGE_SUMMARY.json", change_summary)
    prediction_manifest = {
        "schema_version": "tracking.reconciliation.state8_predictions.v1",
        "video_count": len(generated),
        "methods": [
            "bytetrack_raw",
            "hybrid_bytetrack",
            "realtime_fast",
            "rf_hybrid",
        ],
        "per_video": [
            {
                key: value
                for key, value in row.items()
                if key not in {"final_changes"}
            }
            for row in generated
        ],
        "canonical_sha256": canonical_transfer_hash(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"final_changes"}
                }
                for row in generated
            ]
        ),
    }
    write_json(output_root / "prediction_manifest.json", prediction_manifest)
    no_mp4 = not list(output_root.rglob("*.mp4"))
    if not no_mp4:
        raise State8Error("Unauthorized MP4 artifact created")

    comparisons = {
        "FULL_ACCEPTED_HISTORICAL_OPTIMIZATION_GAIN": {
            metric: (
                metrics["hybrid_bytetrack"][metric]
                - metrics["bytetrack_raw"][metric]
            )
            for metric in metrics["bytetrack_raw"]
        },
        "COMPLETE_CAUSAL_METHOD_COMPARISON": {
            metric: (
                metrics["realtime_fast"][metric]
                - metrics["bytetrack_raw"][metric]
            )
            for metric in metrics["bytetrack_raw"]
        },
        "HYBRID_MECHANISM_TRANSFER_EFFECT": {
            metric: (
                metrics["rf_hybrid"][metric]
                - metrics["realtime_fast"][metric]
            )
            for metric in metrics["realtime_fast"]
        },
    }
    result = {
        "schema_version": "tracking.reconciliation.state8_decision.v1",
        "state": "STATE_8_DEVELOPMENT_EVALUATION",
        "date": DATE,
        "decision": decision,
        "implementation_contract": "PASS",
        "methods": metrics,
        "comparisons": comparisons,
        "rf_hybrid_changes": change_summary,
        "transfer_config_hash": config_hash,
        "two_pass_repeatability": (
            determinism["reevaluation_repeatability"]
        ),
        "input_order_invariance": determinism["input_order_invariance"],
        "identity_conservation": conservation,
        "r0_raw_snapshot_parity": "EXACT_SEMANTIC_ALL_13",
        "transfer_runtime_seconds": transfer_seconds,
        "standard_v2_runtime_seconds": evaluation_seconds,
        "runtime_interpretation": (
            "POST_TRACKLET_TRANSFER_AND_EVALUATOR_ONLY; not complete-method "
            "tracking runtime"
        ),
        "detector_executions": 0,
        "tracker_executions": 0,
        "config_variants_executed": 1,
        "package_variants_executed": 1,
        "post_result_parameter_tuning": 0,
        "per_video_overrides": 0,
        "hand_edited_predictions": 0,
        "current_b1_used": False,
        "unseen_files_accessed": 0,
        "mp4_count": 0,
        "ready_for_state_9_method_freeze": True,
        "ready_for_unseen_evaluation": False,
        "next_state": "STATE_9_METHOD_FREEZE",
    }
    write_json(
        output_root / "STATE_8_DEVELOPMENT_EVALUATION_DECISION.json",
        result,
    )
    run_manifest = load_json(output_root / "run_manifest.json")
    run_manifest.update(
        {
            "status": "COMPLETE",
            "decision": decision,
            "input_inventory_sha256": canonical_transfer_hash(
                input_inventory
            ),
            "prediction_manifest_sha256": sha256_file(
                output_root / "prediction_manifest.json"
            ),
        }
    )
    write_json(output_root / "run_manifest.json", run_manifest)
    inventory = _recursive_inventory(output_root)
    write_json(
        output_root / "artifact_manifest.json",
        {
            "schema_version": "tracking.reconciliation.state8_artifacts.v1",
            "artifact_count": len(inventory),
            "artifacts": inventory,
            "canonical_sha256": canonical_transfer_hash(inventory),
            "recursive_no_mp4": "PASS",
        },
    )
    result["artifact_manifest_sha256"] = sha256_file(
        output_root / "artifact_manifest.json"
    )
    write_json(
        output_root / "STATE_8_DEVELOPMENT_EVALUATION_DECISION.json",
        result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    try:
        execute(
            args.source_repo.resolve(),
            args.worktree_repo.resolve(),
            output_root,
        )
    except Exception as exc:
        manifest_path = output_root / "run_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            manifest.update(
                {
                    "status": "FAILED",
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                }
            )
            write_json(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
