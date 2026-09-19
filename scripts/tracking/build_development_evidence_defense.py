#!/usr/bin/env python3
"""Build the frozen four-method development evidence defense package.

This script reads existing predictions and evaluator artifacts only. It never
invokes a detector, tracker, renderer, or unseen-data discovery path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pig_behavior.evaluation.tracking.cvat_io import (  # noqa: E402
    TrackingObject,
    parse_cvat_video_xml,
)
from pig_behavior.evaluation.tracking.evaluator_standard_v2 import (  # noqa: E402
    aggregate_tracking_standard_v2,
    evaluate_tracking_standard_v2,
)

DATE = "20260730"
EXPECTED_METHODS = (
    "bytetrack_raw",
    "hybrid_bytetrack",
    "realtime_fast",
    "rf_hybrid",
)
EXPECTED_FRAMES = 1800
EXPECTED_VIDEOS = 13
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 10_000
ALPHAS = tuple(round(value * 0.05, 2) for value in range(1, 20))

REQUIRED_AUTHORITIES = (
    Path(
        "docs/tracking/method_standardization/CANONICAL_TRACKING_DEVELOPMENT_RESULTS_20260730.json"
    ),
    Path("docs/tracking/reconciliation/FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json"),
    Path("docs/tracking/rf_hybrid_v2/RF_HYBRID_V2_SCIENTIFIC_DECISION_AUTHORITY_20260730.json"),
    Path("src/pig_behavior/tracking/method_registry.py"),
    Path("outputs/tracking/hybrid_bytetrack_full_20260730_run1/RUN_MANIFEST.json"),
    Path(
        "outputs/tracking/hybrid_bytetrack_full_20260730_run1/"
        "evaluation/iou0_area0_condarea0_merge0/tracking_metrics.csv"
    ),
    Path(
        "outputs/tracking/hybrid_bytetrack_full_20260730_run1/evaluation/"
        "tracking_rule_benchmark_best.json"
    ),
)

PROTECTED_DIRTY = (
    Path(
        "scripts/classification_v2/01_review_units_gui/"
        "review_interaction_blind_calibration_gui_v2.py"
    ),
    Path("scripts/classification_v2/01_review_units_gui/review_temporal_unit_gui.py"),
    Path("tests/test_classification_v2_behavior_gui_readiness.py"),
    Path("tests/test_classification_v2_source_specific_presentation_v2.py"),
)

CANONICAL_RESULTS = REQUIRED_AUTHORITIES[0]
FREEZE_AUTHORITY = REQUIRED_AUTHORITIES[1]
V2_DECISION = REQUIRED_AUTHORITIES[2]
METHOD_REGISTRY = REQUIRED_AUTHORITIES[3]
HYBRID_RERUN_MANIFEST = REQUIRED_AUTHORITIES[4]
HYBRID_RERUN_METRICS = REQUIRED_AUTHORITIES[5]
HYBRID_RERUN_BEST = REQUIRED_AUTHORITIES[6]

LOCKED_POPULATION = Path(
    "outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/"
    "B0_B1_LOCKED_EXECUTION_MANIFEST_20260728.json"
)
B0_PER_VIDEO = Path(
    "outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1/"
    "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv"
)
B0_EPISODES = Path(
    "outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1/"
    "B0_B1_R0_IDENTITY_ERROR_EPISODES.csv"
)
B0_SWAPS = Path(
    "outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1/"
    "B0_B1_R0_PERSISTENT_PAIRWISE_SWAPS.csv"
)
B0_AUTHORITIES = Path(
    "outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1/"
    "B0_B1_R0_IDENTITY_AUTHORITIES.csv"
)
STATE8_ROOT = Path("outputs/tracking/reconciliation_state8_development_20260729_run2")
STATE8_STANDARD = STATE8_ROOT / "standard_v2/pass_1"
STATE8_PER_VIDEO = STATE8_STANDARD / "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv"
STATE8_EPISODES = STATE8_STANDARD / "B0_B1_R0_IDENTITY_ERROR_EPISODES.csv"
STATE8_SWAPS = STATE8_STANDARD / "B0_B1_R0_PERSISTENT_PAIRWISE_SWAPS.csv"
STATE8_AUTHORITIES = STATE8_STANDARD / "B0_B1_R0_IDENTITY_AUTHORITIES.csv"
LEGACY_HYBRID_METRICS = Path(
    "outputs/tracking/historical_h5b_h4_frozen_predictions_20260728/"
    "legacy_evaluation/tracking_metrics.csv"
)

PREDICTION_ROOTS = {
    "bytetrack_raw": Path(
        "outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/"
        "B0_bytetrack_raw/predictions"
    ),
    "hybrid_bytetrack": Path(
        "outputs/tracking/historical_h5b_h4_frozen_predictions_20260728/predictions"
    ),
    "realtime_fast": STATE8_ROOT / "predictions/realtime_fast",
    "rf_hybrid": STATE8_ROOT / "predictions/rf_hybrid",
}

METRIC_COLUMNS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "IDF1",
    "IDP",
    "IDR",
    "IDSW_STANDARD",
    "FP",
    "FN",
    "fragments",
    "wrong_identity_frames",
    "wrong_identity_seconds",
    "recovered_identity_episodes",
    "terminal_identity_episodes",
    "persistent_pairwise_swaps",
)
PAIRED_METRICS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "IDF1",
    "IDSW_STANDARD",
    "wrong_identity_frames",
    "wrong_identity_seconds",
    "terminal_identity_episodes",
    "fragments",
)
BOOTSTRAP_METRICS = (
    "HOTA",
    "AssA",
    "IDF1",
    "wrong_identity_frames",
    "wrong_identity_seconds",
)
LOWER_IS_BETTER = {
    "IDSW_STANDARD",
    "wrong_identity_frames",
    "wrong_identity_seconds",
    "terminal_identity_episodes",
    "fragments",
}

SOURCE_COLUMN_MAP = {
    "hota": "HOTA",
    "deta": "DetA",
    "assa": "AssA",
    "loca": "LocA",
    "idf1": "IDF1",
    "id_precision": "IDP",
    "id_recall": "IDR",
    "idsw_standard": "IDSW_STANDARD",
    "fp": "FP",
    "fn": "FN",
    "fragments": "fragments",
    "wrong_id_matched_frames": "wrong_identity_frames",
    "wrong_id_matched_seconds": "wrong_identity_seconds",
    "recovered_identity_error_episode_count": "recovered_identity_episodes",
    "terminal_identity_error_episode_count": "terminal_identity_episodes",
    "persistent_pairwise_identity_swap_count": "persistent_pairwise_swaps",
}

COMPARISONS = {
    "C1": (
        "realtime_fast",
        "bytetrack_raw",
        "CURRENT_EXECUTABLE_COMPLETE_METHOD_COMPARISON",
    ),
    "C2": (
        "hybrid_bytetrack",
        "realtime_fast",
        "DEVELOPMENT_OFFLINE_VERSUS_CAUSAL_COMPLETE_METHOD_COMPARISON",
    ),
    "C3": (
        "rf_hybrid",
        "realtime_fast",
        "TRANSFER_ABLATION",
    ),
}


class EvidenceError(RuntimeError):
    """Fail-closed development evidence authority error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def collection_hash(records: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in sorted(records, key=lambda item: item["relative_path"])
    ]
    return canonical_hash(normalized)


def file_record(path: Path, base: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"Missing {role}: {path}")
    return {
        "relative_path": path.relative_to(base).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prediction_path(root: Path, method: str, video_id: str) -> Path:
    path = root / PREDICTION_ROOTS[method] / f"{video_id}.xml"
    if not path.is_file():
        raise EvidenceError(f"Missing prediction: {method}/{video_id}")
    return path


def verify_authorities(source_root: Path, worktree: Path) -> dict[str, Any]:
    missing = [str(path) for path in REQUIRED_AUTHORITIES if not (source_root / path).is_file()]
    if missing:
        raise EvidenceError(f"DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: {missing}")
    canonical = read_json(source_root / CANONICAL_RESULTS)
    freeze = read_json(source_root / FREEZE_AUTHORITY)
    v2 = read_json(source_root / V2_DECISION)
    rerun = read_json(source_root / HYBRID_RERUN_BEST)
    canonical_ids = tuple(row["method_id"] for row in canonical["active_method_results"])
    freeze_ids = tuple(freeze["active_methods"])
    v2_ids = tuple(v2["integration_policy"]["active_scientific_methods"])
    if canonical_ids != EXPECTED_METHODS or freeze_ids != EXPECTED_METHODS:
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: method IDs")
    if v2_ids != EXPECTED_METHODS:
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: v2 registry")
    contract = canonical["evaluation_contract"]
    if contract["tracking_evaluator"] != "TRACKING_EVALUATOR_STANDARD_V2":
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: evaluator")
    if contract["identity_episode_evaluator"] != "IDENTITY_ERROR_EPISODES_V2":
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: episodes")
    if contract["include_hidden"] is not True:
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: hidden")
    if int(contract["HOTA_threshold_count"]) != 19:
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: alphas")
    if bool(v2["integration_policy"]["rf_hybrid_v2_promoted"]):
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: v2 active")
    if int(rerun["evaluated_frames"]) != 23_400:
        raise EvidenceError("DEVELOPMENT_EVIDENCE_AUTHORITY_INCOMPLETE: rerun frames")
    main_sha = git_output(source_root, "rev-parse", "HEAD")
    worktree_sha = git_output(worktree, "rev-parse", "HEAD")
    if main_sha != worktree_sha:
        raise EvidenceError("Analysis worktree is not pinned to starting main")
    return {
        "canonical": canonical,
        "freeze": freeze,
        "v2": v2,
        "rerun": rerun,
        "starting_main_sha": main_sha,
        "worktree_sha": worktree_sha,
    }


def canonical_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in payload["active_method_results"]:
        row = {"method_id": source["method_id"], "canonical_version": source["canonical_version"]}
        for column in METRIC_COLUMNS:
            row[column] = source[column]
        row.update(
            {
                "prediction_hash": source["prediction_hash"],
                "GT_hash": source["GT_hash"],
                "evaluator_code_hash": source["evaluator_code_hash"],
                "evaluator_config_hash": source["evaluator_config_hash"],
                "evaluation_population": source["evaluation_population"],
                "include_hidden": source["include_hidden"],
            }
        )
        rows.append(row)
    return rows


def normalize_per_video(
    source_root: Path,
    population: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    b0 = pd.read_csv(source_root / B0_PER_VIDEO)
    state8 = pd.read_csv(source_root / STATE8_PER_VIDEO)
    selected = [b0.loc[b0["arm"] == "B0"].copy()]
    for method in ("hybrid_bytetrack", "realtime_fast", "rf_hybrid"):
        selected.append(state8.loc[state8["arm"] == method].copy())
    frames = pd.concat(selected, ignore_index=True)
    frames.loc[frames["arm"] == "B0", "arm"] = "bytetrack_raw"
    population_by_video = {row["video_key"]: row for row in population}
    rows: list[dict[str, Any]] = []
    for source in frames.to_dict(orient="records"):
        video_id = str(source["video_stem"])
        method = str(source["arm"])
        row: dict[str, Any] = {
            "method_id": method,
            "video_id": video_id,
            "session_key": "UNKNOWN",
            "frame_count": EXPECTED_FRAMES,
            "FPS": 30.0,
            "duration_seconds": EXPECTED_FRAMES / 30.0,
            "evaluation_status": "COMPARABLE_STANDARD_V2_PRIMARY",
        }
        for old, new in SOURCE_COLUMN_MAP.items():
            value = source[old]
            if new in {
                "IDSW_STANDARD",
                "FP",
                "FN",
                "fragments",
                "wrong_identity_frames",
                "recovered_identity_episodes",
                "terminal_identity_episodes",
                "persistent_pairwise_swaps",
            }:
                row[new] = int(value)
            else:
                row[new] = float(value)
        row["mechanism_ranking_eligibility"] = bool(
            population_by_video[video_id]["mechanism_ranking_eligibility"]
        )
        rows.append(row)
    if len(rows) != EXPECTED_VIDEOS * len(EXPECTED_METHODS):
        raise EvidenceError("Per-video table does not contain 52 rows")
    return sorted(rows, key=lambda row: (row["method_id"], row["video_id"]))


def verify_global_counts(
    per_video: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_by_method = {row["method_id"]: row for row in canonical}
    count_columns = (
        "IDSW_STANDARD",
        "FP",
        "FN",
        "fragments",
        "wrong_identity_frames",
        "recovered_identity_episodes",
        "terminal_identity_episodes",
        "persistent_pairwise_swaps",
    )
    checks = []
    for method in EXPECTED_METHODS:
        rows = [row for row in per_video if row["method_id"] == method]
        for column in count_columns:
            observed = sum(int(row[column]) for row in rows)
            expected = int(canonical_by_method[method][column])
            checks.append(
                {
                    "method_id": method,
                    "metric": column,
                    "per_video_sum": observed,
                    "canonical_global": expected,
                    "status": "PASS" if observed == expected else "FAIL",
                }
            )
    failures = [row for row in checks if row["status"] != "PASS"]
    if failures:
        raise EvidenceError(f"Per-video/global count mismatch: {failures}")
    return {
        "status": "PASS",
        "checks": checks,
        "nonadditive_metrics": ["HOTA", "DetA", "AssA", "LocA", "IDF1", "IDP", "IDR"],
        "explanation": (
            "HOTA combines per-alpha sufficient statistics and IDF1 recomputes "
            "from summed IDTP/IDFP/IDFN; neither is an arithmetic mean of videos."
        ),
    }


def build_input_authority(
    source_root: Path,
    output_dir: Path,
    verified: dict[str, Any],
    population: list[dict[str, Any]],
) -> dict[str, Any]:
    registry_hash = sha256_file(source_root / METHOD_REGISTRY)
    authority_records = [
        file_record(source_root / path, source_root, "REQUIRED_AUTHORITY")
        for path in REQUIRED_AUTHORITIES
    ]
    gt_records = []
    prediction_authorities: dict[str, Any] = {}
    for video in population:
        gt_path = Path(video["gt_path"])
        gt_records.append(
            {
                "video_id": video["video_key"],
                "path": str(gt_path),
                "sha256": sha256_file(gt_path),
                "manifest_sha256": video["gt_sha256"],
                "frame_count": video["expected_frame_count"],
                "FPS": 30.0,
            }
        )
    canonical_by_method = {
        row["method_id"]: row for row in verified["canonical"]["active_method_results"]
    }
    for method in EXPECTED_METHODS:
        records = []
        for video in population:
            path = prediction_path(source_root, method, video["video_key"])
            records.append(file_record(path, source_root, "PREDICTION_XML"))
        prediction_authorities[method] = {
            "authority_path": str(PREDICTION_ROOTS[method]),
            "authority_hash": canonical_by_method[method]["prediction_hash"],
            "raw_file_collection_sha256": collection_hash(records),
            "xml_count": len(records),
            "files": records,
        }
    protected = []
    for relative in PROTECTED_DIRTY:
        path = source_root / relative
        protected.append(file_record(path, source_root, "UNRELATED_PROTECTED"))
    payload = {
        "schema_version": "tracking.development_evidence.input_authority.v1",
        "date": DATE,
        "starting_main_sha": verified["starting_main_sha"],
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "dataset-contract-leakage-guard",
            "safe-refactor-test-guardian",
        ],
        "active_methods": list(EXPECTED_METHODS),
        "method_registry": {
            "path": str(METHOD_REGISTRY),
            "sha256": registry_hash,
        },
        "prediction_authorities": prediction_authorities,
        "ground_truth_authorities": gt_records,
        "ground_truth_collection_sha256": canonical_hash(gt_records),
        "evaluation_contract": verified["canonical"]["evaluation_contract"],
        "method_comparability": {
            "same_frame_set": "PASS_0_TO_1799_ALL_13",
            "same_include_hidden": "PASS_TRUE_PRIMARY",
            "same_GT_version": "PASS",
            "same_matching_eligibility": "PASS_TRACKING_MATCHING_STANDARD_V2",
            "same_area_filtering": "PASS_NO_METHOD_SPECIFIC_AREA_FILTER",
            "same_HOTA_thresholds": list(ALPHAS),
            "same_IDSW_definition": "PASS_IDSW_STANDARD",
            "status": "COMPARABLE",
        },
        "hybrid_rerun": {
            "role": "SUPPLEMENTARY_METRIC_LEVEL_NEAR_PARITY_EVIDENCE",
            "exact_prediction_reproducibility": "NOT_ESTABLISHED",
            "metric_level_near_parity_reproducibility": "ESTABLISHED",
            "manifest": authority_records[4],
            "metrics": authority_records[5],
            "best_summary": authority_records[6],
        },
        "required_authority_records": authority_records,
        "protected_dirty_files": protected,
        "unseen_files_accessed": 0,
        "prediction_authority_created": False,
    }
    write_json(
        output_dir / "DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json",
        payload,
    )
    return payload


def build_session_map(
    output_dir: Path,
    population: list[dict[str, Any]],
) -> None:
    rows = []
    for video in population:
        video_id = video["video_key"]
        rows.append(
            {
                "video_id": video_id,
                "video_path_or_authority_id": video["source_video_path"],
                "frame_count": video["expected_frame_count"],
                "duration_seconds": video["expected_frame_count"] / 30.0,
                "FPS": 30.0,
                "recording_timestamp_if_proven": "NOT_PROVEN",
                "camera_id_if_proven": "NOT_PROVEN",
                "pen_id_if_proven": "NOT_PROVEN",
                "session_key": "UNKNOWN",
                "session_key_basis": "NO_DOCUMENTED_SESSION_METADATA",
                "same_session_confidence": "UNKNOWN",
                "independent_cluster_key": video_id,
                "notes": (
                    "Filename tokens were not treated as proof of session, camera, "
                    "pen, or independence."
                ),
            }
        )
    fields = (
        "video_id",
        "video_path_or_authority_id",
        "frame_count",
        "duration_seconds",
        "FPS",
        "recording_timestamp_if_proven",
        "camera_id_if_proven",
        "pen_id_if_proven",
        "session_key",
        "session_key_basis",
        "same_session_confidence",
        "independent_cluster_key",
        "notes",
    )
    write_csv(output_dir / "DEVELOPMENT_VIDEO_SESSION_MAP_20260730.csv", rows, fields)
    write_json(
        output_dir / "DEVELOPMENT_INDEPENDENCE_AUDIT_20260730.json",
        {
            "schema_version": "tracking.development_evidence.independence.v1",
            "TOTAL_VIDEOS": EXPECTED_VIDEOS,
            "TOTAL_FRAMES": EXPECTED_VIDEOS * EXPECTED_FRAMES,
            "PROVEN_SESSION_COUNT": 0,
            "PROVEN_INDEPENDENT_CLUSTERS": "UNKNOWN",
            "UNRESOLVED_SESSION_GROUPS": EXPECTED_VIDEOS,
            "STATISTICAL_CLUSTER_UNIT": "VIDEO",
            "STATISTICAL_CLUSTER_INTERPRETATION": (
                "DESCRIPTIVE_VIDEO_CLUSTER; inter-video independence unresolved"
            ),
            "PSEUDOREPLICATION_RISK": "MAJOR",
            "FRAME_LEVEL_INFERENCE_ALLOWED": "NO",
            "PER_SESSION_ANALYSIS": "NOT_JUSTIFIED",
        },
    )


def build_per_video(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "method_id",
        "video_id",
        "session_key",
        "frame_count",
        "duration_seconds",
        "FPS",
        *METRIC_COLUMNS,
        "evaluation_status",
    )
    write_csv(
        output_dir / "DEVELOPMENT_PER_VIDEO_TRACKING_METRICS_20260730.csv",
        rows,
        fields,
    )


def build_paired(
    output_dir: Path,
    per_video: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["method_id"], row["video_id"]): row for row in per_video}
    video_ids = sorted({row["video_id"] for row in per_video})
    paired_rows = []
    summary_rows = []
    loco_rows = []
    for comparison_id, (candidate, reference, interpretation) in COMPARISONS.items():
        for metric in PAIRED_METRICS:
            metric_rows = []
            for video_id in video_ids:
                candidate_value = float(lookup[(candidate, video_id)][metric])
                reference_value = float(lookup[(reference, video_id)][metric])
                difference = candidate_value - reference_value
                improvement = -difference if metric in LOWER_IS_BETTER else difference
                tie_tolerance = 0.0 if metric in LOWER_IS_BETTER else 1e-12
                if abs(improvement) <= tie_tolerance:
                    result = "TIE"
                elif improvement > 0:
                    result = "WIN"
                else:
                    result = "LOSS"
                row = {
                    "comparison_id": comparison_id,
                    "comparison_type": interpretation,
                    "candidate_method": candidate,
                    "reference_method": reference,
                    "cluster_unit": "VIDEO",
                    "cluster_id": video_id,
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "reference_value": reference_value,
                    "paired_difference": difference,
                    "improvement_oriented_difference": improvement,
                    "tie_tolerance": tie_tolerance,
                    "result": result,
                }
                paired_rows.append(row)
                metric_rows.append(row)
            differences = [float(row["paired_difference"]) for row in metric_rows]
            wins = sum(row["result"] == "WIN" for row in metric_rows)
            ties = sum(row["result"] == "TIE" for row in metric_rows)
            losses = sum(row["result"] == "LOSS" for row in metric_rows)
            full_mean = statistics.mean(differences)
            omitted_means = []
            for omitted in video_ids:
                retained = [
                    float(row["paired_difference"])
                    for row in metric_rows
                    if row["cluster_id"] != omitted
                ]
                mean_without = statistics.mean(retained)
                omitted_means.append((omitted, mean_without))
                loco_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "candidate_method": candidate,
                        "reference_method": reference,
                        "metric": metric,
                        "resampling_unit": "VIDEO",
                        "omitted_cluster": omitted,
                        "cluster_count_retained": len(retained),
                        "full_mean_paired_difference": full_mean,
                        "leave_one_out_mean_paired_difference": mean_without,
                        "change_from_full_mean": mean_without - full_mean,
                        "independence_status": "INTER_VIDEO_INDEPENDENCE_UNRESOLVED",
                    }
                )
            influential = max(
                omitted_means,
                key=lambda item: abs(item[1] - full_mean),
            )
            summary_rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_type": interpretation,
                    "candidate_method": candidate,
                    "reference_method": reference,
                    "metric": metric,
                    "cluster_count": len(metric_rows),
                    "mean_paired_difference": full_mean,
                    "median_paired_difference": statistics.median(differences),
                    "minimum_paired_difference": min(differences),
                    "maximum_paired_difference": max(differences),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "proportion_clusters_improved": wins / len(metric_rows),
                    "leave_one_cluster_out_min": min(value for _, value in omitted_means),
                    "leave_one_cluster_out_max": max(value for _, value in omitted_means),
                    "most_influential_cluster": influential[0],
                    "influence_change_from_full_mean": influential[1] - full_mean,
                    "consistency_interpretation": (
                        "CONSISTENT_DIRECTION"
                        if losses == 0 and wins > 0
                        else "MIXED_ACROSS_VIDEOS"
                    ),
                }
            )
    paired_fields = tuple(paired_rows[0])
    summary_fields = tuple(summary_rows[0])
    loco_fields = tuple(loco_rows[0])
    write_csv(
        output_dir / "DEVELOPMENT_PAIRED_COMPARISONS_20260730.csv",
        paired_rows,
        paired_fields,
    )
    write_csv(
        output_dir / "DEVELOPMENT_WIN_TIE_LOSS_SUMMARY_20260730.csv",
        summary_rows,
        summary_fields,
    )
    write_csv(
        output_dir / "DEVELOPMENT_LEAVE_ONE_CLUSTER_OUT_20260730.csv",
        loco_rows,
        loco_fields,
    )
    return paired_rows, summary_rows, loco_rows


def build_bootstrap(
    output_dir: Path,
    paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows = []
    for comparison_id in COMPARISONS:
        for metric in BOOTSTRAP_METRICS:
            values = [
                float(row["paired_difference"])
                for row in paired_rows
                if row["comparison_id"] == comparison_id and row["metric"] == metric
            ]
            samples = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                samples.append(statistics.mean(rng.choice(values) for _ in range(len(values))))
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "resampling_unit": "VIDEO",
                    "resample_count": BOOTSTRAP_RESAMPLES,
                    "CI_method": "PERCENTILE_DESCRIPTIVE",
                    "cluster_count": len(values),
                    "mean_paired_difference": statistics.mean(values),
                    "bootstrap_2_5_percentile": percentile(samples, 0.025),
                    "bootstrap_97_5_percentile": percentile(samples, 0.975),
                    "formal_CI_status": ("INSUFFICIENT_INDEPENDENT_CLUSTERS_FOR_RELIABLE_CI"),
                    "reason": (
                        "Thirteen video units exist, but recording-session "
                        "independence is not proven."
                    ),
                }
            )
    write_csv(
        output_dir / "DEVELOPMENT_CLUSTER_BOOTSTRAP_SUMMARY_20260730.csv",
        rows,
        tuple(rows[0]),
    )
    interpretation = """# Development uncertainty interpretation

The resampling unit is the video, never the frame. The 10,000-resample paired
bootstrap uses seed `20260730` and reports descriptive percentile ranges.

These ranges are not claimed as robust confidence intervals because available
authorities do not prove that the 13 videos are independent recording-session
clusters. Leave-one-video-out results are therefore reported alongside every
comparison, and no `p < 0.05` decision is made.

Status: `INSUFFICIENT_INDEPENDENT_CLUSTERS_FOR_RELIABLE_CI`.
"""
    (output_dir / "DEVELOPMENT_UNCERTAINTY_INTERPRETATION_20260730.md").write_text(
        interpretation,
        encoding="utf-8",
        newline="\n",
    )
    return rows


def visible_metric_row(method: str, scope: str, evaluation: Any) -> dict[str, Any]:
    metrics = asdict(evaluation.metrics)
    return {
        "method_id": method,
        "evaluation_scope": scope,
        "HOTA": metrics["hota"],
        "DetA": metrics["deta"],
        "AssA": metrics["assa"],
        "LocA": metrics["loca"],
        "IDF1": metrics["idf1"],
        "IDSW_STANDARD": metrics["idsw_standard"],
        "wrong_identity_frames": metrics["wrong_id_matched_frames"],
        "wrong_identity_seconds": metrics["wrong_id_matched_seconds"],
        "fragments": metrics["fragments"],
        "terminal_identity_episodes": metrics["terminal_identity_error_episode_count"],
        "prediction_hash": "BOUND_BY_INPUT_AUTHORITY",
        "GT_hash": "675cf37c4f924e391ffa457ba6c6e9453b967af318f37ecd2bc1ab1190a1d9dd",
        "evaluator_hash": metrics["evaluator_code_sha"],
    }


def build_hidden_sensitivity(
    source_root: Path,
    output_dir: Path,
    population: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    per_video: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    canonical_by_method = {row["method_id"]: row for row in canonical}
    primary_by_video = {(row["method_id"], row["video_id"]): row for row in per_video}
    rows = []
    per_video_rows = []
    visible_evaluations: dict[str, list[Any]] = {}
    for method in EXPECTED_METHODS:
        evaluations = []
        for video in population:
            video_id = video["video_key"]
            gt = parse_cvat_video_xml(
                Path(video["gt_path"]),
                include_hidden=False,
                start_frame=0,
                end_frame=EXPECTED_FRAMES - 1,
            )
            pred = parse_cvat_video_xml(
                prediction_path(source_root, method, video_id),
                include_hidden=False,
                start_frame=0,
                end_frame=EXPECTED_FRAMES - 1,
            )
            evaluations.append(
                evaluate_tracking_standard_v2(
                    gt,
                    pred,
                    video_stem=video_id,
                    include_hidden=False,
                    detection_iou_threshold=0.5,
                    frames_per_second=30.0,
                    evaluator_code_sha=(canonical_by_method[method]["evaluator_code_hash"]),
                )
            )
        visible_evaluations[method] = evaluations
        for video, evaluation in zip(population, evaluations, strict=True):
            video_id = video["video_key"]
            visible_video = visible_metric_row(
                method,
                "VISIBLE_ONLY_SENSITIVITY",
                evaluation,
            )
            primary_video = primary_by_video[(method, video_id)]
            per_video_rows.append(
                {
                    "method_id": method,
                    "video_id": video_id,
                    "HOTA_primary": primary_video["HOTA"],
                    "HOTA_visible_only": visible_video["HOTA"],
                    "HOTA_visible_minus_primary": (visible_video["HOTA"] - primary_video["HOTA"]),
                    "AssA_primary": primary_video["AssA"],
                    "AssA_visible_only": visible_video["AssA"],
                    "AssA_visible_minus_primary": (visible_video["AssA"] - primary_video["AssA"]),
                    "IDF1_primary": primary_video["IDF1"],
                    "IDF1_visible_only": visible_video["IDF1"],
                    "IDF1_visible_minus_primary": (visible_video["IDF1"] - primary_video["IDF1"]),
                    "wrong_identity_frames_primary": primary_video["wrong_identity_frames"],
                    "wrong_identity_frames_visible_only": visible_video["wrong_identity_frames"],
                    "wrong_identity_frames_visible_minus_primary": (
                        visible_video["wrong_identity_frames"]
                        - primary_video["wrong_identity_frames"]
                    ),
                    "interpretation": ("SENSITIVITY_ONLY_NOT_HUMAN_HIDDEN_VALIDATION"),
                }
            )
        aggregate = aggregate_tracking_standard_v2(evaluations)
        visible = visible_metric_row(method, "VISIBLE_ONLY_SENSITIVITY", aggregate)
        visible["prediction_hash"] = canonical_by_method[method]["prediction_hash"]
        visible["GT_hash"] = canonical_by_method[method]["GT_hash"]
        visible["evaluator_hash"] = canonical_by_method[method]["evaluator_code_hash"]
        rows.append(visible)
        primary = {
            "method_id": method,
            "evaluation_scope": "PRIMARY_INCLUDE_HIDDEN",
            "HOTA": canonical_by_method[method]["HOTA"],
            "DetA": canonical_by_method[method]["DetA"],
            "AssA": canonical_by_method[method]["AssA"],
            "LocA": canonical_by_method[method]["LocA"],
            "IDF1": canonical_by_method[method]["IDF1"],
            "IDSW_STANDARD": canonical_by_method[method]["IDSW_STANDARD"],
            "wrong_identity_frames": canonical_by_method[method]["wrong_identity_frames"],
            "wrong_identity_seconds": canonical_by_method[method]["wrong_identity_seconds"],
            "fragments": canonical_by_method[method]["fragments"],
            "terminal_identity_episodes": canonical_by_method[method]["terminal_identity_episodes"],
            "prediction_hash": canonical_by_method[method]["prediction_hash"],
            "GT_hash": canonical_by_method[method]["GT_hash"],
            "evaluator_hash": canonical_by_method[method]["evaluator_code_hash"],
        }
        rows.append(primary)
    fields = tuple(rows[0])
    write_csv(
        output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_RESULTS_20260730.csv",
        rows,
        fields,
    )
    write_csv(
        output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_PER_VIDEO_20260730.csv",
        per_video_rows,
        tuple(per_video_rows[0]),
    )
    primary_rank = [
        row["method_id"]
        for row in sorted(
            (row for row in rows if row["evaluation_scope"] == "PRIMARY_INCLUDE_HIDDEN"),
            key=lambda row: float(row["HOTA"]),
            reverse=True,
        )
    ]
    visible_rank = [
        row["method_id"]
        for row in sorted(
            (row for row in rows if row["evaluation_scope"] == "VISIBLE_ONLY_SENSITIVITY"),
            key=lambda row: float(row["HOTA"]),
            reverse=True,
        )
    ]
    conclusion = (
        "RANKING_ROBUST_TO_HIDDEN_POLICY"
        if primary_rank == visible_rank
        else "METHOD_RANKING_SENSITIVE_TO_HIDDEN_POLICY"
    )
    most_sensitive = {}
    for method in EXPECTED_METHODS:
        candidates = [row for row in per_video_rows if row["method_id"] == method]
        top = max(
            candidates,
            key=lambda row: abs(float(row["HOTA_visible_minus_primary"])),
        )
        most_sensitive[method] = {
            "video_id": top["video_id"],
            "HOTA_visible_minus_primary": top["HOTA_visible_minus_primary"],
        }
    write_json(
        output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_DECISION_20260730.json",
        {
            "primary_HOTA_ranking": primary_rank,
            "visible_only_HOTA_ranking": visible_rank,
            "conclusion": conclusion,
            "largest_absolute_HOTA_sensitivity_by_method": most_sensitive,
            "primary_authority_unchanged": True,
            "interpretation": "Secondary sensitivity only; not a replacement evaluator.",
        },
    )
    return rows, visible_evaluations


def tracklet_statistics(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    lengths = []
    xml_tracks = 0
    for track in root.findall(".//track"):
        frames = sorted(
            int(box.get("frame", "0"))
            for box in track.findall("box")
            if box.get("outside", "0") != "1"
        )
        if not frames:
            continue
        xml_tracks += 1
        current = 1
        previous = frames[0]
        for frame in frames[1:]:
            if frame == previous + 1:
                current += 1
            else:
                lengths.append(current)
                current = 1
            previous = frame
        lengths.append(current)
    return {
        "xml_track_count": xml_tracks,
        "predicted_tracklet_count": len(lengths),
        "tracklet_lengths_frames": lengths,
        "minimum_tracklet_length_frames": min(lengths) if lengths else 0,
        "median_tracklet_length_frames": statistics.median(lengths) if lengths else 0,
        "short_tracklets_lt_15_frames": sum(value < 15 for value in lengths),
        "short_tracklets_15_to_59_frames": sum(15 <= value < 60 for value in lengths),
        "tracklets_60_to_299_frames": sum(60 <= value < 300 for value in lengths),
        "tracklets_ge_300_frames": sum(value >= 300 for value in lengths),
    }


def build_fragmentation(
    source_root: Path,
    output_dir: Path,
    population: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    per_video: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_by_method = {row["method_id"]: row for row in canonical}
    gt_identity_count = sum(
        int(tracklet_statistics(Path(video["gt_path"]))["xml_track_count"]) for video in population
    )
    if gt_identity_count <= 0:
        raise EvidenceError("Frozen GT population has no identities")
    rows = []
    for method in EXPECTED_METHODS:
        per_method = [row for row in per_video if row["method_id"] == method]
        stats = [
            tracklet_statistics(prediction_path(source_root, method, video["video_key"]))
            for video in population
        ]
        total_tracks = sum(int(row["xml_track_count"]) for row in stats)
        total_tracklets = sum(int(row["predicted_tracklet_count"]) for row in stats)
        all_min = min(int(row["minimum_tracklet_length_frames"]) for row in stats)
        all_lengths = [int(length) for row in stats for length in row["tracklet_lengths_frames"]]
        rows.append(
            {
                "method_id": method,
                "predicted_xml_track_count": total_tracks,
                "predicted_tracklet_count": total_tracklets,
                "GT_identity_count": gt_identity_count,
                "predicted_tracklets_per_GT_identity": (total_tracklets / float(gt_identity_count)),
                "GT_identities_matched_to_more_than_one_predicted_tracklet": (
                    "NOT_AVAILABLE_UNDER_FROZEN_SUMMARY_AUTHORITY"
                ),
                "median_tracklet_length_frames": statistics.median(all_lengths),
                "minimum_tracklet_length_frames": all_min,
                "short_tracklets_lt_15_frames": sum(
                    int(row["short_tracklets_lt_15_frames"]) for row in stats
                ),
                "short_tracklets_15_to_59_frames": sum(
                    int(row["short_tracklets_15_to_59_frames"]) for row in stats
                ),
                "tracklets_60_to_299_frames": sum(
                    int(row["tracklets_60_to_299_frames"]) for row in stats
                ),
                "tracklets_ge_300_frames": sum(
                    int(row["tracklets_ge_300_frames"]) for row in stats
                ),
                "fragments": canonical_by_method[method]["fragments"],
                "fragments_per_GT_identity": (
                    canonical_by_method[method]["fragments"] / float(gt_identity_count)
                ),
                "mostly_tracked": "NOT_AVAILABLE",
                "partially_tracked": "NOT_AVAILABLE",
                "mostly_lost": "NOT_AVAILABLE",
                "identity_continuity_through_occlusion_episodes": ("NOT_AVAILABLE"),
                "track_termination_before_identity_recovery": (
                    "NOT_IDENTIFIABLE_FROM_FRAGMENT_COUNT_ALONE"
                ),
                "missing_detection_spans": "REPRESENTED_BY_STANDARD_V2_FRAGMENTS",
                "IDSW_STANDARD": canonical_by_method[method]["IDSW_STANDARD"],
                "wrong_identity_frames": canonical_by_method[method]["wrong_identity_frames"],
                "terminal_identity_episodes": canonical_by_method[method][
                    "terminal_identity_episodes"
                ],
                "maximum_video_fragments": max(int(row["fragments"]) for row in per_method),
                "top_three_video_fragment_share": sum(
                    sorted((int(row["fragments"]) for row in per_method), reverse=True)[:3]
                )
                / float(canonical_by_method[method]["fragments"]),
            }
        )
    write_csv(
        output_dir / "DEVELOPMENT_FRAGMENTATION_AND_COMPLETENESS_AUDIT_20260730.csv",
        rows,
        tuple(rows[0]),
    )
    hybrid_per_video = [row for row in per_video if row["method_id"] == "hybrid_bytetrack"]
    legacy = pd.read_csv(source_root / LEGACY_HYBRID_METRICS)
    legacy = legacy.loc[legacy["video_stem"] != "ALL"]
    strict_legacy = int(legacy["fragments"].sum())
    gap_tolerant = int(legacy["gap_tolerant_fragments"].sum())
    defense = {
        "schema_version": "tracking.development_evidence.hybrid_idsw_zero.v1",
        "conclusion": "IDSW_ZERO_SUPPORTED_BY_BROAD_IDENTITY_CONTINUITY",
        "evidence": {
            "IDSW_STANDARD": 0,
            "wrong_identity_frames": 24,
            "terminal_identity_episodes": 0,
            "persistent_pairwise_swaps": 0,
            "IDF1": 0.9915010683760683,
            "AssA": 0.9119035496598343,
            "fragments": 425,
            "FP": 1579,
            "FN": 1579,
            "per_video_IDF1_min": min(float(row["IDF1"]) for row in hybrid_per_video),
            "per_video_IDF1_median": statistics.median(
                float(row["IDF1"]) for row in hybrid_per_video
            ),
            "per_video_AssA_min": min(float(row["AssA"]) for row in hybrid_per_video),
            "per_video_AssA_median": statistics.median(
                float(row["AssA"]) for row in hybrid_per_video
            ),
            "standard_v2_fragment_top_three_share": sorted(
                (int(row["fragments"]) for row in hybrid_per_video),
                reverse=True,
            )[:3],
            "legacy_same_prediction_strict_fragments": strict_legacy,
            "legacy_same_prediction_gap_tolerant_fragments_15": gap_tolerant,
        },
        "interpretation": (
            "Zero IDSW is not interpreted alone. High global and per-video "
            "identity scores, 24 transient wrong-ID animal-frames, no terminal "
            "episodes, and no persistent swaps support continuity. The large "
            "standard fragment count is primarily a detection/localization-gap "
            "signal; the supplementary 15-frame gap-tolerant count is small."
        ),
        "limitation": (
            "The gap-tolerant fragment statistic comes from the same historical "
            "prediction artifact under the legacy supplementary continuity report, "
            "not from the Standard V2 headline metric."
        ),
    }
    write_json(output_dir / "DEVELOPMENT_HYBRID_IDSW_ZERO_DEFENSE_20260730.json", defense)
    return defense


def load_episode_tables(source_root: Path) -> pd.DataFrame:
    b0 = pd.read_csv(source_root / B0_EPISODES)
    b0 = b0.loc[b0["arm"] == "B0"].copy()
    b0["arm"] = "bytetrack_raw"
    state8 = pd.read_csv(source_root / STATE8_EPISODES)
    state8 = state8.loc[state8["arm"].isin(EXPECTED_METHODS[1:])].copy()
    return pd.concat([b0, state8], ignore_index=True)


def load_swap_tables(source_root: Path) -> pd.DataFrame:
    b0 = pd.read_csv(source_root / B0_SWAPS)
    b0 = b0.loc[b0["arm"] == "B0"].copy()
    b0["arm"] = "bytetrack_raw"
    state8 = pd.read_csv(source_root / STATE8_SWAPS)
    state8 = state8.loc[state8["arm"].isin(EXPECTED_METHODS[1:])].copy()
    return pd.concat([b0, state8], ignore_index=True)


def load_authority_tables(source_root: Path) -> pd.DataFrame:
    b0 = pd.read_csv(source_root / B0_AUTHORITIES)
    b0 = b0.loc[b0["arm"] == "B0"].copy()
    b0["arm"] = "bytetrack_raw"
    state8 = pd.read_csv(source_root / STATE8_AUTHORITIES)
    state8 = state8.loc[state8["arm"].isin(EXPECTED_METHODS[1:])].copy()
    return pd.concat([b0, state8], ignore_index=True)


def parse_row_keys(value: str) -> list[tuple[str, int, str, str]]:
    parsed = ast.literal_eval(value)
    return [(str(a), int(b), str(c), str(d)) for a, b, c, d in parsed]


def object_lookup(path: Path) -> dict[tuple[int, str], TrackingObject]:
    parsed = parse_cvat_video_xml(
        path,
        include_hidden=True,
        start_frame=0,
        end_frame=EXPECTED_FRAMES - 1,
    )
    return {(frame, item.obj_id): item for frame, objects in parsed.items() for item in objects}


def bbox_text(item: TrackingObject | None) -> str:
    return json.dumps(list(item.bbox)) if item is not None else "NOT_AVAILABLE"


def build_audit_pack(
    source_root: Path,
    output_dir: Path,
    population: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    episodes = load_episode_tables(source_root)
    swaps = load_swap_tables(source_root)
    authorities = load_authority_tables(source_root)
    population_by_video = {row["video_key"]: row for row in population}
    gt_cache: dict[str, dict[tuple[int, str], TrackingObject]] = {}
    pred_cache: dict[tuple[str, str], dict[tuple[int, str], TrackingObject]] = {}
    audit: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_wrong_row(
        method: str,
        episode_id: str,
        key: tuple[str, int, str, str],
        reason: str,
        category: str = "WRONG_IDENTITY_MATCH",
    ) -> None:
        video_id, frame, gt_id, pred_id = key
        if video_id not in gt_cache:
            gt_cache[video_id] = object_lookup(Path(population_by_video[video_id]["gt_path"]))
        pred_key = (method, video_id)
        if pred_key not in pred_cache:
            pred_cache[pred_key] = object_lookup(prediction_path(source_root, method, video_id))
        gt_obj = gt_cache[video_id].get((frame, gt_id))
        pred_obj = pred_cache[pred_key].get((frame, pred_id))
        key_id = (method, video_id, str(frame), episode_id)
        existing = audit.get(key_id)
        if existing:
            if reason not in existing["selection_reason"]:
                existing["selection_reason"] += f"; {reason}"
            return
        audit[key_id] = {
            "audit_item_id": f"AUDIT-{len(audit) + 1:04d}",
            "video_id": video_id,
            "frame_or_span": str(frame),
            "method_id": method,
            "GT_identity": gt_id,
            "predicted_identity": pred_id,
            "GT_bbox": bbox_text(gt_obj),
            "predicted_bbox": bbox_text(pred_obj),
            "Hidden_status": ("YES" if gt_obj is not None and gt_obj.hidden else "NO_OR_UNKNOWN"),
            "matching_eligibility": "AUTHORITATIVE_ELIGIBLE_MATCH_IOU_0.50",
            "error_category": category,
            "episode_id": episode_id,
            "metric_contribution": "one wrong-ID animal-frame",
            "selection_reason": reason,
            "visual_review_status": "NOT_REVIEWED",
            "reviewer_comment": "",
        }

    hybrid = episodes.loc[episodes["arm"] == "hybrid_bytetrack"]
    for row in hybrid.to_dict(orient="records"):
        for key in parse_row_keys(str(row["row_keys"])):
            add_wrong_row(
                "hybrid_bytetrack",
                str(row["event_id"]),
                key,
                "ALL_24_HYBRID_WRONG_ID_FRAMES; ALL_HYBRID_EPISODES",
            )

    realtime_terminal = episodes.loc[
        (episodes["arm"] == "realtime_fast") & (episodes["status"] == "terminal")
    ]
    for row in realtime_terminal.to_dict(orient="records"):
        add_wrong_row(
            "realtime_fast",
            str(row["event_id"]),
            parse_row_keys(str(row["row_keys"]))[0],
            "ALL_REALTIME_FAST_TERMINAL_EPISODES",
            "TERMINAL_IDENTITY_EPISODE",
        )

    raw_longest = (
        episodes.loc[episodes["arm"] == "bytetrack_raw"]
        .sort_values(
            "duration_frames",
            ascending=False,
        )
        .head(8)
    )
    for row in raw_longest.to_dict(orient="records"):
        add_wrong_row(
            "bytetrack_raw",
            str(row["event_id"]),
            parse_row_keys(str(row["row_keys"]))[0],
            "LONGEST_BYTETRACK_RAW_WRONG_ID_EPISODE",
        )

    for method in EXPECTED_METHODS:
        recovered = episodes.loc[
            (episodes["arm"] == method) & (episodes["status"] == "recovered")
        ].sort_values("duration_frames")
        if recovered.empty:
            continue
        sample = pd.concat([recovered.head(1), recovered.tail(1)]).drop_duplicates(
            subset=["event_id"]
        )
        for row in sample.to_dict(orient="records"):
            add_wrong_row(
                method,
                str(row["event_id"]),
                parse_row_keys(str(row["row_keys"]))[0],
                "STRATIFIED_RECOVERED_EPISODE_SAMPLE",
            )

    for row in swaps.to_dict(orient="records"):
        video_id = str(row["sequence_key"])
        frame = int(row["start_frame"])
        item_id = (str(row["arm"]), video_id, f"swap-{frame}", str(row["event_id"]))
        audit[item_id] = {
            "audit_item_id": f"AUDIT-{len(audit) + 1:04d}",
            "video_id": video_id,
            "frame_or_span": f"{row['start_frame']}-{row['end_frame']}",
            "method_id": row["arm"],
            "GT_identity": row["gt_ids"],
            "predicted_identity": "RECIPROCAL_PAIRWISE_SWAP",
            "GT_bbox": "EPISODE_LEVEL",
            "predicted_bbox": "EPISODE_LEVEL",
            "Hidden_status": "MIXED_OR_UNKNOWN",
            "matching_eligibility": "AUTHORITATIVE_PAIRWISE_EVENT",
            "error_category": "PERSISTENT_PAIRWISE_SWAP",
            "episode_id": row["event_id"],
            "metric_contribution": "one persistent pairwise swap event",
            "selection_reason": "ALL_PERSISTENT_SWAPS_EVERY_ACTIVE_METHOD",
            "visual_review_status": "NOT_REVIEWED",
            "reviewer_comment": "",
        }

    authority_lookup = {
        (str(row["arm"]), str(row["sequence_key"]), str(row["gt_id"])): str(row["pred_id"])
        for row in authorities.to_dict(orient="records")
    }
    for method in EXPECTED_METHODS:
        for video in population:
            video_id = video["video_key"]
            if video_id not in gt_cache:
                gt_cache[video_id] = object_lookup(Path(video["gt_path"]))
            hidden_items = [item for item in gt_cache[video_id].values() if item.hidden]
            if not hidden_items:
                continue
            gt_obj = sorted(hidden_items, key=lambda item: (item.frame, item.obj_id))[0]
            pred_id = authority_lookup.get((method, video_id, gt_obj.obj_id), "NOT_MAPPED")
            pred_key = (method, video_id)
            if pred_key not in pred_cache:
                pred_cache[pred_key] = object_lookup(prediction_path(source_root, method, video_id))
            pred_obj = pred_cache[pred_key].get((gt_obj.frame, pred_id))
            item_id = (method, video_id, f"hidden-{gt_obj.frame}", gt_obj.obj_id)
            audit[item_id] = {
                "audit_item_id": f"AUDIT-{len(audit) + 1:04d}",
                "video_id": video_id,
                "frame_or_span": str(gt_obj.frame),
                "method_id": method,
                "GT_identity": gt_obj.obj_id,
                "predicted_identity": pred_id,
                "GT_bbox": bbox_text(gt_obj),
                "predicted_bbox": bbox_text(pred_obj),
                "Hidden_status": "YES",
                "matching_eligibility": ("PRIMARY_INCLUDED; VISIBLE_ONLY_EXCLUDED_BEFORE_MATCHING"),
                "error_category": "HIDDEN_POLICY_SENSITIVITY_REFERENCE",
                "episode_id": "NOT_APPLICABLE",
                "metric_contribution": "population membership changes by policy",
                "selection_reason": "HIDDEN_STATUS_MATERIALLY_CHANGES_ELIGIBILITY",
                "visual_review_status": "NOT_REVIEWED",
                "reviewer_comment": "",
            }

    for comparison in COMPARISONS:
        summary = next(
            row
            for row in summary_rows
            if row["comparison_id"] == comparison and row["metric"] == "HOTA"
        )
        item_id = (comparison, str(summary["most_influential_cluster"]), "influence", "HOTA")
        audit[item_id] = {
            "audit_item_id": f"AUDIT-{len(audit) + 1:04d}",
            "video_id": summary["most_influential_cluster"],
            "frame_or_span": "0-1799",
            "method_id": f"{summary['candidate_method']}_vs_{summary['reference_method']}",
            "GT_identity": "ALL",
            "predicted_identity": "ALL",
            "GT_bbox": "NOT_APPLICABLE",
            "predicted_bbox": "NOT_APPLICABLE",
            "Hidden_status": "MIXED",
            "matching_eligibility": "WHOLE_VIDEO_STANDARD_V2",
            "error_category": "INFLUENTIAL_CLUSTER",
            "episode_id": "NOT_APPLICABLE",
            "metric_contribution": "largest leave-one-video HOTA mean influence",
            "selection_reason": "CONTRIBUTES_MOST_TO_PAIRED_HOTA_DIFFERENCE",
            "visual_review_status": "NOT_REVIEWED",
            "reviewer_comment": "",
        }
    rows = list(audit.values())
    fields = (
        "audit_item_id",
        "video_id",
        "frame_or_span",
        "method_id",
        "GT_identity",
        "predicted_identity",
        "GT_bbox",
        "predicted_bbox",
        "Hidden_status",
        "matching_eligibility",
        "error_category",
        "episode_id",
        "metric_contribution",
        "selection_reason",
        "visual_review_status",
        "reviewer_comment",
    )
    write_csv(
        output_dir / "DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv",
        rows,
        fields,
    )
    write_json(
        output_dir / "DEVELOPMENT_GT_ERROR_REVIEW_BUNDLE_20260730.json",
        {
            "schema_version": "tracking.development_evidence.review_bundle.v1",
            "item_count": len(rows),
            "source_csv": "DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv",
            "rendered_video_or_frame_artifacts": 0,
            "MP4_created": 0,
            "HUMAN_GT_AUDIT_COMPLETED": "NO",
            "review_policy": "Review references in place; do not edit GT or predictions.",
        },
    )
    hybrid_wrong = {
        (row["video_id"], row["frame_or_span"], row["GT_identity"])
        for row in rows
        if row["method_id"] == "hybrid_bytetrack"
        and "ALL_24_HYBRID_WRONG_ID_FRAMES" in row["selection_reason"]
    }
    if len(hybrid_wrong) != 24:
        raise EvidenceError(f"Hybrid audit pack has {len(hybrid_wrong)} wrong rows")
    return rows


def write_metric_specification(output_dir: Path) -> None:
    specification = """# Development identity diagnostic metric specification

This document restates, without changing, `TRACKING_EVALUATOR_STANDARD_V2`,
`TRACKING_MATCHING_STANDARD_V2`, and `IDENTITY_ERROR_EPISODES_V2`.

## Population and GT–prediction matching

Outside boxes are excluded. `include_hidden` is applied symmetrically before
generic matching. The primary study uses `include_hidden=true`. Detection,
CLEAR continuity, and episode rows use IoU 0.50 eligibility-constrained
assignment: ineligible edges are removed first, then eligible cardinality is
maximized before total IoU. HOTA separately follows TrackEval's prescribed
global-alignment assignment at each of 19 alphas from 0.05 through 0.95.

## Identity authority and wrong-ID rows

Identity global metrics use one sequence-local TrackEval Identity assignment.
Episode severity instead freezes `IDENTITY_AUTHORITY_FIRST_OBSERVATION_V2`:
the first unambiguous eligible match binds an unmapped prediction identity to
an unmapped GT identity and never changes. A matched resolved GT row is wrong
when the observed prediction identity is not the frozen expected identity.
One video frame can contribute multiple wrong animal-frame observations. An
unmatched GT or prediction is not a wrong-ID row; it contributes FN or FP.

Wrong-ID seconds sum `1/FPS` per wrong animal-frame, using each video's proven
positive FPS. Thus the unit is animal-seconds, not elapsed wall-clock span.

## Episodes

The primary key is `(sequence_key, gt_id)`. Wrong rows join when no correct
matched row intervenes and the frame delta is at most 15. Unmatched rows add no
wrong exposure and do not recover an episode. A later authoritative correct
match makes the episode recovered. An episode is terminal when it contains the
GT trajectory's final authoritative matched observation and has no recovery.
Recovered and terminal are mutually exclusive. A gap larger than 15 splits an
episode and censors the earlier component unless later correct evidence proves
recovery.

## Pairwise swaps, Hidden, fragments, and edge cases

A pairwise swap requires reciprocal wrong ownership of two GT identities in
the same frame. Unordered pairs are counted once. Persistence requires at least
60 direct joint observations, or two terminal linked GT episodes that still
target each other at their final authoritative observations. Three-way cycles
are not pairwise swaps.

With `include_hidden=false`, Hidden rows are excluded before matching and
cannot create matches, misses, fragments, switches, or episodes. A fragment is
a resolved GT trajectory returning to a matched state after one or more
authoritative evaluated GT observations were unmatched. Missing predictions
may create FN and fragmentation, but not wrong-ID exposure by themselves.

Tied first identity authority is retained as ambiguous and excluded from
authoritative severity totals. Duplicate identities in one frame, invalid
boxes, non-positive FPS, malformed authority maps, and mixed sequence keys fail
closed. Every sequence resets authority, switch memory, fragments, and episode
state. A global permutation of prediction labels is harmless when the frozen
one-to-one mapping is consistent throughout the sequence.
"""
    (output_dir / "DEVELOPMENT_IDENTITY_DIAGNOSTIC_METRIC_SPECIFICATION_20260730.md").write_text(
        specification,
        encoding="utf-8",
        newline="\n",
    )
    golden = {
        "schema_version": "tracking.identity_diagnostic_golden_examples.v1",
        "contract_ids": [
            "TRACKING_EVALUATOR_STANDARD_V2",
            "TRACKING_MATCHING_STANDARD_V2",
            "IDENTITY_ERROR_EPISODES_V2",
        ],
        "examples": [
            {"id": "correct_continuity", "sequence": ["A", "A", "A"], "idsw": 0, "wrong_rows": 0},
            {"id": "one_frame_switch", "sequence": ["A", "B", "A"], "idsw": 2, "wrong_rows": 1},
            {
                "id": "recovered_switch",
                "sequence": ["A", "B", "B", "A"],
                "idsw": 2,
                "wrong_rows": 2,
                "recovered": 1,
            },
            {
                "id": "terminal_wrong_owner",
                "sequence": ["A", "B", "B"],
                "idsw": 1,
                "wrong_rows": 2,
                "terminal": 1,
            },
            {
                "id": "fragmentation_without_switch",
                "sequence": ["A", None, "A"],
                "idsw": 0,
                "wrong_rows": 0,
                "fragments": 1,
            },
            {
                "id": "hidden_interval",
                "sequence": ["A", "B", "A"],
                "hidden_indices": [1],
                "include_hidden_true_idsw": 2,
                "include_hidden_false_idsw": 0,
            },
            {
                "id": "prediction_gap",
                "sequence": ["A", None, None, "A"],
                "idsw": 0,
                "wrong_rows": 0,
            },
            {
                "id": "global_identity_permutation",
                "gt": ["A", "B"],
                "prediction": ["pB", "pA"],
                "consistent": True,
                "wrong_rows": 0,
            },
            {
                "id": "simultaneous_pairwise_swap",
                "gt_A": ["pA", "pB"],
                "gt_B": ["pB", "pA"],
                "wrong_rows": 2,
                "pairwise_event": 1,
            },
        ],
    }
    write_json(
        output_dir / "DEVELOPMENT_IDENTITY_DIAGNOSTIC_GOLDEN_EXAMPLES_20260730.json",
        golden,
    )


def build_fairness(output_dir: Path, canonical: list[dict[str, Any]]) -> None:
    common = {
        "evaluation_population": "FROZEN_13_VIDEO_DEVELOPMENT_POPULATION",
        "evaluator_contract": "TRACKING_EVALUATOR_STANDARD_V2",
    }
    method_rows = [
        {
            "row_type": "METHOD",
            "method_or_comparison": "bytetrack_raw",
            "detector_producer": "LIVE_YOLO_TRACK",
            "detector_cadence": "EVERY_FRAME",
            "detection_threshold": 0.25,
            "maximum_detections": 32,
            "tracker_core": "ULTRALYTICS_BYTETRACK_RAW",
            "tracker_state_lifecycle": "ONE_INSTANCE_PER_VIDEO_PERSIST_TRUE",
            "causal_future_frame_policy": "CAUSAL_NONE",
            "offline_processing": "NO_ACCEPTED_HYBRID_REPAIRS",
            "prediction_authority_type": "FROZEN_EXECUTABLE_PREDICTION_SET",
            "execution_authority_status": "ESTABLISHED",
            "runtime_authority_status": "PROTOCOL_FROZEN_NOT_EXECUTED",
            **common,
            "comparison_classification": "NOT_APPLICABLE",
            "allowed_wording": "Current executable ByteTrack baseline.",
            "forbidden_wording": "Original historical raw result.",
        },
        {
            "row_type": "METHOD",
            "method_or_comparison": "hybrid_bytetrack",
            "detector_producer": "HISTORICAL_LIVE_YOLO_TRACK",
            "detector_cadence": "EVERY_FRAME",
            "detection_threshold": 0.20,
            "maximum_detections": 64,
            "tracker_core": "FULL_ACCEPTED_HYBRID_BYTETRACK_LINEAGE",
            "tracker_state_lifecycle": "PERSISTENT_PER_VIDEO_PLUS_POST_VIDEO",
            "causal_future_frame_policy": "OFFLINE_FUTURE_ALLOWED",
            "offline_processing": "FULL_21_STAGE_ACCEPTED_LINEAGE",
            "prediction_authority_type": "SURVIVING_HISTORICAL_FINAL_XML_SET",
            "execution_authority_status": "ALGORITHMIC_LINEAGE_RECOVERED",
            "runtime_authority_status": "RUNTIME_PROVENANCE_INCOMPLETE",
            **common,
            "comparison_classification": "NOT_APPLICABLE",
            "allowed_wording": "Historical offline development champion.",
            "forbidden_wording": "ByteTrack plus one generic repair.",
        },
        {
            "row_type": "METHOD",
            "method_or_comparison": "realtime_fast",
            "detector_producer": "YOLO_PREDICT",
            "detector_cadence": "EVERY_SECOND_FRAME",
            "detection_threshold": 0.25,
            "maximum_detections": 32,
            "tracker_core": "CAUSAL_RF_ASSOCIATION",
            "tracker_state_lifecycle": "ONE_CAUSAL_STATE_PER_VIDEO",
            "causal_future_frame_policy": "CAUSAL_ZERO_DELAY",
            "offline_processing": "NONE",
            "prediction_authority_type": "FROZEN_EXECUTABLE_PREDICTION_SET",
            "execution_authority_status": "ESTABLISHED",
            "runtime_authority_status": "PROTOCOL_FROZEN_NOT_EXECUTED",
            **common,
            "comparison_classification": "NOT_APPLICABLE",
            "allowed_wording": "Causal realtime-oriented complete method.",
            "forbidden_wording": "Empirically established real-time system.",
        },
        {
            "row_type": "METHOD",
            "method_or_comparison": "rf_hybrid",
            "detector_producer": "SHARED_REALTIME_FAST_EVIDENCE",
            "detector_cadence": "EVERY_SECOND_FRAME",
            "detection_threshold": 0.25,
            "maximum_detections": 32,
            "tracker_core": "FROZEN_RF_THEN_TRANSFER_SUBSET",
            "tracker_state_lifecycle": "CAUSAL_CORE_PLUS_POST_VIDEO_TRANSFER",
            "causal_future_frame_policy": "OFFLINE_TRANSFER_ALLOWED",
            "offline_processing": "PREDECLARED_10_STAGE_TRANSFER",
            "prediction_authority_type": "FROZEN_DEVELOPMENT_ABLATION",
            "execution_authority_status": "TRANSFER_SIGNAL_MIXED",
            "runtime_authority_status": "NOT_DEPLOYMENT_ELIGIBLE",
            **common,
            "comparison_classification": "NOT_APPLICABLE",
            "allowed_wording": "Mixed transfer ablation.",
            "forbidden_wording": "Quality or deployment upgrade.",
        },
    ]
    comparison_rows = [
        (
            "realtime_fast versus bytetrack_raw",
            "COMPLETE_METHOD_COMPARISON",
            "Quality comparison between complete executable methods.",
            "Pure association-core superiority.",
        ),
        (
            "hybrid_bytetrack versus realtime_fast",
            "COMPLETE_METHOD_COMPARISON_WITH_HISTORICAL_RUNTIME_LIMITATION",
            "Development offline-versus-causal complete-method comparison.",
            "Runtime-controlled or topology-matched ablation.",
        ),
        (
            "rf_hybrid versus realtime_fast",
            "TRANSFER_ABLATION",
            "Predeclared mechanism-transfer effect with mixed outcome.",
            "Overall improvement based only on lower IDSW.",
        ),
    ]
    rows = list(method_rows)
    for name, classification, allowed, forbidden in comparison_rows:
        rows.append(
            {
                "row_type": "COMPARISON",
                "method_or_comparison": name,
                "detector_producer": "SEE_METHOD_ROWS",
                "detector_cadence": "SEE_METHOD_ROWS",
                "detection_threshold": "SEE_METHOD_ROWS",
                "maximum_detections": "SEE_METHOD_ROWS",
                "tracker_core": "DIFFERENT_COMPLETE_GRAPHS",
                "tracker_state_lifecycle": "SEE_METHOD_ROWS",
                "causal_future_frame_policy": "SEE_METHOD_ROWS",
                "offline_processing": "SEE_METHOD_ROWS",
                "prediction_authority_type": "SEE_METHOD_ROWS",
                "execution_authority_status": "SEE_METHOD_ROWS",
                "runtime_authority_status": "SEE_METHOD_ROWS",
                **common,
                "comparison_classification": classification,
                "allowed_wording": allowed,
                "forbidden_wording": forbidden,
            }
        )
    write_csv(
        output_dir / "DEVELOPMENT_COMPLETE_METHOD_FAIRNESS_MATRIX_20260730.csv",
        rows,
        tuple(rows[0]),
    )


def build_baseline_runtime(output_dir: Path) -> None:
    write_json(
        output_dir / "DEVELOPMENT_BASELINE_ADEQUACY_ASSESSMENT_20260730.json",
        {
            "schema_version": "tracking.development_evidence.baseline_adequacy.v1",
            "intended_paper_emphasis": "BEHAVIOR_PIPELINE_PRIMARY",
            "decision": "CURRENT_BASELINE_SUFFICIENT_FOR_UPSTREAM_MODULE_STUDY",
            "ByteTrack_appropriate_conventional_baseline": True,
            "internal_variants_are_external_baselines": False,
            "standalone_novel_tracking_claim_supported": False,
            "additional_MOT_baseline_likely_requested_for_tracking_primary_scope": True,
            "defensible_scope": (
                "Tracking as an evaluated upstream component of the pig-behavior pipeline."
            ),
            "later_work_if_tracking_becomes_primary": (
                "Freeze and execute at least one additional external MOT baseline "
                "under the same 13+12-video contract."
            ),
        },
    )
    write_json(
        output_dir / "DEVELOPMENT_RUNTIME_CLAIM_AUDIT_20260730.json",
        {
            "schema_version": "tracking.development_evidence.runtime_claim.v1",
            "runtime_protocol": (
                "docs/tracking/method_standardization/"
                "TRACKING_RUNTIME_BENCHMARK_PROTOCOL_20260730.json"
            ),
            "protocol_status": "FROZEN_PROTOCOL_NOT_EXECUTED",
            "methods": {
                "bytetrack_raw": {
                    "causal": True,
                    "future_frames_used": False,
                    "detector_cadence": "EVERY_FRAME",
                    "measured_end_to_end_FPS_available": False,
                    "latency_distribution_available": False,
                    "hardware_authority_available": True,
                    "peak_GPU_memory_available": False,
                    "repeat_count_available": False,
                    "IO_policy_available": True,
                    "runtime_claim_status": "RUNTIME_EVIDENCE_INCOMPLETE",
                },
                "realtime_fast": {
                    "causal": True,
                    "future_frames_used": False,
                    "detector_cadence": "EVERY_SECOND_FRAME",
                    "measured_end_to_end_FPS_available": False,
                    "latency_distribution_available": False,
                    "hardware_authority_available": True,
                    "peak_GPU_memory_available": False,
                    "repeat_count_available": False,
                    "IO_policy_available": True,
                    "runtime_claim_status": "REALTIME_ORIENTED_BUT_NOT_YET_BENCHMARKED",
                },
            },
            "recommended_wording": "causal realtime-oriented method",
            "forbidden_wording": "real-time system",
        },
    )


def build_claims(output_dir: Path, hidden_conclusion: str) -> list[dict[str, Any]]:
    authority = "CANONICAL_TRACKING_DEVELOPMENT_RESULTS_20260730.json"
    rows = [
        (
            "C01",
            "hybrid_bytetrack has the best development HOTA",
            "hybrid_bytetrack",
            "STRONGLY_SUPPORTED_ON_DEVELOPMENT",
            "Hybrid had the highest HOTA on the 13-video development population.",
            "Hybrid is universally best.",
            "Same population was used for development and evaluation.",
            "Independent 12-video evaluation",
            "YES",
            "Results",
        ),
        (
            "C02",
            "hybrid_bytetrack has the strongest development identity continuity",
            "hybrid_bytetrack",
            "SUPPORTED_WITH_MAJOR_LIMITATION",
            "Hybrid had IDSW=0, 24 wrong-ID frames, no terminal episodes, "
            "and high IDF1 on development.",
            "Identity continuity generalizes.",
            "Fragmentation and historical runtime limitations require disclosure.",
            "Unseen identity analysis",
            "YES",
            "Results/Limitations",
        ),
        (
            "C03",
            "hybrid_bytetrack is best in every tracking dimension",
            "hybrid_bytetrack",
            "NOT_SUPPORTED",
            "No such claim is allowed.",
            "Best in every tracking dimension.",
            "Hybrid has higher FP/FN/fragments than executable methods.",
            "None; revise claim",
            "NO",
            "Discussion",
        ),
        (
            "C04",
            "realtime_fast improves over current bytetrack_raw",
            "C1",
            "SUPPORTED_WITH_MAJOR_LIMITATION",
            "Realtime_fast improves several development quality and identity "
            "metrics as a complete method.",
            "Its association core alone is superior.",
            "Detector cadence and full topology differ; clusters are development-only.",
            "Independent 12-video comparison",
            "YES",
            "Results",
        ),
        (
            "C05",
            "realtime_fast has a better association core than ByteTrack",
            "C1",
            "NOT_SUPPORTED",
            "Only a complete-method comparison is supported.",
            "Better association core.",
            "Detector cadence, producer, and topology differ.",
            "Controlled component ablation",
            "NO",
            "Methods",
        ),
        (
            "C06",
            "realtime_fast is real-time",
            "realtime_fast",
            "REQUIRES_RUNTIME_BENCHMARK",
            "Describe it as causal and realtime-oriented.",
            "Real-time system.",
            "Frozen benchmark protocol has not been executed.",
            "Paired frozen runtime benchmark",
            "NO",
            "Runtime",
        ),
        (
            "C07",
            "rf_hybrid improves realtime_fast",
            "C3",
            "NOT_SUPPORTED",
            "rf_hybrid produced a mixed transfer result, not an overall improvement.",
            "rf_hybrid improves realtime_fast.",
            "Lower IDSW coexists with worse HOTA, IDF1, wrong-ID exposure, and terminal episodes.",
            "No retuning in this study",
            "NO",
            "Ablation",
        ),
        (
            "C08",
            "rf_hybrid reduces IDSW but increases wrong-ID exposure",
            "C3",
            "STRONGLY_SUPPORTED_ON_DEVELOPMENT",
            "rf_hybrid reduced IDSW from 29 to 18 but increased wrong-ID frames "
            "from 11,893 to 14,515.",
            "Lower IDSW proves better identity quality.",
            "Development population only.",
            "Independent 12-video evaluation if retained",
            "YES",
            "Ablation",
        ),
        (
            "C09",
            "the hybrid result is reproducible",
            "hybrid_bytetrack",
            "SUPPORTED_WITH_MAJOR_LIMITATION",
            "A current full rerun established metric-level near parity, not exact "
            "prediction reproduction.",
            "Byte-exact historical reproduction.",
            "Historical runtime provenance remains incomplete.",
            "Exact archival runtime evidence",
            "NO",
            "Reproducibility",
        ),
        (
            "C10",
            "the results generalize to unseen sessions",
            "all",
            "REQUIRES_UNSEEN",
            "No generalization claim is made yet.",
            "Generalizes to unseen sessions.",
            "No independent unseen evaluation exists.",
            "Frozen 12-video evaluation",
            "YES",
            "Limitations",
        ),
        (
            "C11",
            "the methods are robust to Hidden-policy changes",
            "all",
            "DESCRIPTIVE_ONLY",
            (
                "The development HOTA ordering remained unchanged under the "
                f"visible-only sensitivity: {hidden_conclusion}."
            ),
            "Robust to all visibility annotation uncertainty.",
            "Hidden values are tracker-derived and visible-only changes the evaluated population.",
            "Human Hidden review and unseen sensitivity",
            "YES",
            "Sensitivity",
        ),
        (
            "C12",
            "the four-method study proves deployment readiness",
            "all",
            "NOT_SUPPORTED",
            "The study supplies development evidence only.",
            "Deployment ready.",
            "No unseen validation and no frozen runtime benchmark.",
            "Unseen plus runtime benchmark",
            "YES",
            "Limitations",
        ),
    ]
    output = [
        {
            "claim_id": row[0],
            "proposed_claim": row[1],
            "method_or_comparison": row[2],
            "supporting_authority": authority,
            "support_level": row[3],
            "allowed_wording": row[4],
            "forbidden_wording": row[5],
            "known_limitation": row[6],
            "remaining_test_needed": row[7],
            "unseen_required": row[8],
            "paper_section": row[9],
        }
        for row in rows
    ]
    write_csv(
        output_dir / "DEVELOPMENT_CLAIM_LIMITATION_MATRIX_20260730.csv",
        output,
        tuple(output[0]),
    )
    return output


def build_objections(
    output_dir: Path,
    hidden_conclusion: str,
    hybrid_defense: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = [
        (
            "DEVELOPMENT_OVERFITTING",
            "CRITICAL",
            "Methods were developed and assessed on the same 13 videos.",
            "Frozen metrics and no post-result tuning.",
            "Paired per-video and influence analysis.",
            "Development effects are descriptive, not confirmatory.",
            "Independent 12-video evidence remains absent.",
            "Report development results with explicit hypothesis-generating scope.",
            "YES",
            "12-video evaluation",
            "NO",
        ),
        (
            "PSEUDOREPLICATION",
            "MAJOR",
            "23,400 frames are temporally dependent.",
            "Sequence-local evaluator and 13 per-video rows.",
            "Video-cluster bootstrap and leave-one-video-out analysis.",
            "No frame-level inference was used.",
            "Video independence is not proven.",
            "Use video as the descriptive unit and avoid frame-level CIs.",
            "YES",
            "Session metadata",
            "NO",
        ),
        (
            "SESSION_NONINDEPENDENCE",
            "MAJOR",
            "Several videos may share recording conditions.",
            "No proven session metadata was found.",
            "Session map records every grouping as unknown.",
            "Per-session aggregation was not justified.",
            "Independent session count remains unknown.",
            "Disclose unresolved session dependence.",
            "YES",
            "Proven recording-session metadata",
            "NO",
        ),
        (
            "AGGREGATE_DOMINATION",
            "MAJOR",
            "Aggregate metrics can hide influential videos.",
            "Mandatory per-video Standard V2 metrics.",
            "Win/tie/loss, paired differences, and leave-one-video-out ranges.",
            "Influential videos are identified explicitly.",
            "Thirteen videos remain a small development population.",
            "Present per-video distributions beside global metrics.",
            "YES",
            "Independent larger set",
            "NO",
        ),
        (
            "IDSW_GAMING_BY_FRAGMENTATION",
            "MAJOR",
            "Zero IDSW can arise if difficult tracks terminate.",
            "Hybrid IDF1/AssA, errors, fragments, and episode tables.",
            "Tracklet, gap-tolerant fragment, terminal, and persistent-swap audit.",
            hybrid_defense["conclusion"],
            "Mostly-tracked breakdown is unavailable.",
            "Report IDSW jointly with IDF1, AssA, FP/FN, fragments, wrong-ID "
            "exposure, and terminal episodes.",
            "YES",
            "Human trajectory audit",
            "NO",
        ),
        (
            "DETECTION_ASSOCIATION_TRADEOFF",
            "MODERATE",
            "Hybrid has higher FP/FN/fragments despite stronger identity.",
            "Canonical DetA/AssA/LocA and raw counts.",
            "Per-video detection-association comparison.",
            "The trade-off is real and is not hidden.",
            "Causal and offline methods have different topology.",
            "Describe hybrid as identity-strong, not best in every dimension.",
            "YES",
            "Unseen comparison",
            "NO",
        ),
        (
            "METHOD_FAIRNESS",
            "MAJOR",
            "Complete pipelines use different producers and cadence.",
            "Method registry and fairness matrix.",
            "All pairwise claims were classified by design.",
            "Only complete-method and transfer claims are allowed.",
            "No pure core ablation exists.",
            "State detector cadence and future-frame policy in the comparison table.",
            "YES",
            "Controlled ablation if core claim is desired",
            "NO",
        ),
        (
            "HISTORICAL_RUNTIME_INCOMPLETENESS",
            "MAJOR",
            "Historical hybrid predictions are not byte-exactly reproducible.",
            "Historical XML authority plus full rerun.",
            "Historical-versus-rerun metric table.",
            "Metric-level near parity is established; exact parity is not.",
            "Exact historical package/runtime provenance is unavailable.",
            "Retain historical XMLs as primary and label rerun supplementary.",
            "YES",
            "Exact archival runtime evidence",
            "NO",
        ),
        (
            "HIDDEN_GT_UNCERTAINTY",
            "MAJOR",
            "Hidden labels are tracker-derived and may be uncertain.",
            "Primary include-Hidden and evaluator contract.",
            "Visible-only frozen-prediction sensitivity.",
            hidden_conclusion,
            "Visible-only excludes rather than corrects uncertain Hidden labels.",
            "Keep include-Hidden primary and report visible-only sensitivity.",
            "YES",
            "Human Hidden review",
            "NO",
        ),
        (
            "CUSTOM_METRIC_VALIDITY",
            "MODERATE",
            "Wrong-ID episodes are custom diagnostics.",
            "Versioned Standard V2 contracts and tests.",
            "Specification plus machine-readable golden cases.",
            "Definitions and units are explicit; diagnostics supplement standard metrics.",
            "External adoption is not established.",
            "Lead with HOTA/IDF1/IDSW and label episode metrics diagnostic.",
            "YES",
            "Optional independent implementation",
            "NO",
        ),
        (
            "REALTIME_CLAIM",
            "MAJOR",
            "Causal semantics do not prove throughput or latency.",
            "Prefix invariance and frozen runtime protocol.",
            "Runtime authority inventory.",
            "Realtime_fast is causal and realtime-oriented, not empirically benchmarked here.",
            "No paired p50/p95/FPS/memory authority.",
            "Use 'causal realtime-oriented method'.",
            "YES",
            "Frozen runtime benchmark",
            "NO",
        ),
        (
            "BASELINE_INSUFFICIENCY",
            "MODERATE",
            "Only ByteTrack is an external executable baseline.",
            "Current ByteTrack authority and internal methods.",
            "Baseline adequacy assessment.",
            "Sufficient for an upstream behavior-pipeline study; not for a tracking-primary claim.",
            "Additional MOT baseline absent.",
            "Narrow the paper claim or add a baseline later.",
            "YES",
            "External MOT baseline for tracking-primary scope",
            "NO",
        ),
        (
            "NO_UNSEEN_GENERALIZATION",
            "CRITICAL",
            "No independent test currently supports generalization.",
            "Development-only freeze.",
            "Claim matrix explicitly blocks generalization.",
            "No unseen claim is made.",
            "Twelve-video evaluation is pending.",
            "Call all current results development evidence.",
            "YES",
            "Frozen 12-video evaluation",
            "YES_FOR_GENERALIZATION_ONLY",
        ),
    ]
    rows = [
        {
            "objection_id": row[0],
            "likely_reviewer_objection": row[2],
            "severity": row[1],
            "why_the_objection_is_reasonable": row[2],
            "current_evidence": row[3],
            "new_analysis_performed": row[4],
            "result": row[5],
            "residual_limitation": row[6],
            "defensible_response": row[7],
            "paper_change_required": row[8],
            "future_evidence_required": row[9],
            "fatal_to_current_development_claims": row[10],
        }
        for row in entries
    ]
    write_csv(
        output_dir / "DEVELOPMENT_REVIEWER_OBJECTION_RESPONSE_MATRIX_20260730.csv",
        rows,
        tuple(rows[0]),
    )
    return rows


def build_reproducibility_table(
    output_dir: Path,
    canonical: list[dict[str, Any]],
    rerun: dict[str, Any],
) -> None:
    historical = next(row for row in canonical if row["method_id"] == "hybrid_bytetrack")
    mapping = {
        "HOTA": "hota",
        "DetA": "deta",
        "AssA": "assa",
        "LocA": "loca",
        "IDF1": "idf1",
        "IDSW_STANDARD": "idsw_standard",
        "FP": "fp",
        "FN": "fn",
        "fragments": "fragments",
        "wrong_identity_frames": "wrong_id_matched_frames",
        "wrong_identity_seconds": "wrong_id_matched_seconds",
        "recovered_identity_episodes": "recovered_identity_error_episode_count",
        "terminal_identity_episodes": "terminal_identity_error_episode_count",
        "persistent_pairwise_swaps": "persistent_pairwise_identity_swap_count",
    }
    rows = []
    for metric, key in mapping.items():
        history_value = historical[metric]
        rerun_value = rerun[key]
        rows.append(
            {
                "metric": metric,
                "historical_authority_value": history_value,
                "latest_full_rerun_value": rerun_value,
                "absolute_difference": float(rerun_value) - float(history_value),
                "exact_metric_match": history_value == rerun_value,
                "interpretation": "SUPPLEMENTARY_NEAR_PARITY_NOT_BYTE_EXACT",
            }
        )
    write_csv(
        output_dir / "PAPER_TABLE_HYBRID_REPRODUCIBILITY_20260730.csv",
        rows,
        tuple(rows[0]),
    )


def build_paper_tables(
    output_dir: Path,
    canonical: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    per_video: list[dict[str, Any]],
    loco: list[dict[str, Any]],
) -> None:
    canonical_fields = (
        "method_id",
        "canonical_version",
        *METRIC_COLUMNS,
        "prediction_hash",
        "evaluation_population",
        "include_hidden",
    )
    write_csv(
        output_dir / "PAPER_TABLE_CANONICAL_DEVELOPMENT_RESULTS_20260730.csv",
        canonical,
        canonical_fields,
    )
    write_csv(
        output_dir / "PAPER_TABLE_PER_VIDEO_PAIRED_DIFFERENCES_20260730.csv",
        paired,
        tuple(paired[0]),
    )
    write_csv(
        output_dir / "PAPER_TABLE_HIDDEN_SENSITIVITY_20260730.csv",
        hidden,
        tuple(hidden[0]),
    )
    write_csv(
        output_dir / "PAPER_TABLE_CLAIMS_AND_LIMITATIONS_20260730.csv",
        claims,
        tuple(claims[0]),
    )
    hota = [row for row in paired if row["metric"] == "HOTA"]
    idf1 = [row for row in paired if row["metric"] == "IDF1"]
    write_csv(
        output_dir / "FIGURE_DATA_PER_VIDEO_HOTA_DELTAS_20260730.csv",
        hota,
        tuple(hota[0]),
    )
    write_csv(
        output_dir / "FIGURE_DATA_PER_VIDEO_IDF1_DELTAS_20260730.csv",
        idf1,
        tuple(idf1[0]),
    )
    exposure = [
        {
            "method_id": row["method_id"],
            "video_id": row["video_id"],
            "wrong_identity_frames": row["wrong_identity_frames"],
            "wrong_identity_seconds": row["wrong_identity_seconds"],
            "terminal_identity_episodes": row["terminal_identity_episodes"],
        }
        for row in per_video
    ]
    write_csv(
        output_dir / "FIGURE_DATA_WRONG_ID_EXPOSURE_BY_VIDEO_20260730.csv",
        exposure,
        tuple(exposure[0]),
    )
    fragmentation = [
        {
            "method_id": row["method_id"],
            "video_id": row["video_id"],
            "fragments": row["fragments"],
            "HOTA": row["HOTA"],
            "AssA": row["AssA"],
            "IDF1": row["IDF1"],
            "IDSW_STANDARD": row["IDSW_STANDARD"],
            "wrong_identity_frames": row["wrong_identity_frames"],
        }
        for row in per_video
    ]
    write_csv(
        output_dir / "FIGURE_DATA_FRAGMENTATION_VS_IDENTITY_QUALITY_20260730.csv",
        fragmentation,
        tuple(fragmentation[0]),
    )
    write_csv(
        output_dir / "FIGURE_DATA_LEAVE_ONE_CLUSTER_OUT_20260730.csv",
        loco,
        tuple(loco[0]),
    )


def build_report(
    output_dir: Path,
    canonical: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    hidden_conclusion: str,
    hybrid_defense: dict[str, Any],
    claims: list[dict[str, Any]],
    objections: list[dict[str, Any]],
    rerun: dict[str, Any],
) -> None:
    by_method = {row["method_id"]: row for row in canonical}
    hidden_global = pd.read_csv(output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_RESULTS_20260730.csv")
    hidden_lookup = {
        (str(row["method_id"]), str(row["evaluation_scope"])): row
        for row in hidden_global.to_dict(orient="records")
    }
    hybrid_primary_advantage = float(
        hidden_lookup[("hybrid_bytetrack", "PRIMARY_INCLUDE_HIDDEN")]["HOTA"]
    ) - float(hidden_lookup[("realtime_fast", "PRIMARY_INCLUDE_HIDDEN")]["HOTA"])
    hybrid_visible_advantage = float(
        hidden_lookup[("hybrid_bytetrack", "VISIBLE_ONLY_SENSITIVITY")]["HOTA"]
    ) - float(hidden_lookup[("realtime_fast", "VISIBLE_ONLY_SENSITIVITY")]["HOTA"])
    realtime_primary_wrong = int(
        hidden_lookup[("realtime_fast", "PRIMARY_INCLUDE_HIDDEN")]["wrong_identity_frames"]
    )
    realtime_visible_wrong = int(
        hidden_lookup[("realtime_fast", "VISIBLE_ONLY_SENSITIVITY")]["wrong_identity_frames"]
    )
    hidden_per_video = pd.read_csv(
        output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_PER_VIDEO_20260730.csv"
    )
    hidden_per_video["absolute_HOTA_delta"] = hidden_per_video["HOTA_visible_minus_primary"].abs()
    top_hidden_rows = hidden_per_video.sort_values(
        "absolute_HOTA_delta",
        ascending=False,
    ).head(4)
    top_hidden_text = "\n".join(
        f"- {row.method_id}:{row.video_id} ({row.HOTA_visible_minus_primary:+.4f})"
        for row in top_hidden_rows.itertuples()
    )
    c1_hota = next(
        row for row in summaries if row["comparison_id"] == "C1" and row["metric"] == "HOTA"
    )
    c2_hota = next(
        row for row in summaries if row["comparison_id"] == "C2" and row["metric"] == "HOTA"
    )
    c3_hota = next(
        row for row in summaries if row["comparison_id"] == "C3" and row["metric"] == "HOTA"
    )
    strong = sum(row["support_level"] == "STRONGLY_SUPPORTED_ON_DEVELOPMENT" for row in claims)
    major = sum(row["support_level"] == "SUPPORTED_WITH_MAJOR_LIMITATION" for row in claims)
    unseen = sum(row["support_level"] == "REQUIRES_UNSEEN" for row in claims)
    unsupported = sum(row["support_level"] == "NOT_SUPPORTED" for row in claims)
    text = f"""# Tracking development evidence and reviewer defense report

## 1. Executive summary

### Development evidence

- Strongly established: on the frozen 13-video development population,
  `hybrid_bytetrack` has the highest HOTA ({by_method["hybrid_bytetrack"]["HOTA"]:.6f})
  and strongest identity diagnostics; `rf_hybrid` is a mixed negative transfer
  result rather than a quality upgrade.
- Descriptively observed: per-video paired effects, leave-one-video-out
  influence, and video-cluster bootstrap ranges are reported without treating
  frames as independent.
- Not established: byte-exact historical hybrid reproducibility, empirical
  real-time performance, a pure association-core effect, deployment readiness,
  or sufficiency for a tracking-method-primary paper.

### Generalization evidence not yet available

No claim is made about unseen recordings or sessions. The planned separate
12-video evaluation is required for every generalization statement.

## 2. Study scope

This package analyzes existing predictions only. Detector runs, tracker runs,
parameter tuning, per-video rules, MP4 generation, unseen access, and prediction
or GT modification are all zero. The 23,400 frames are measurement rows, not
independent inferential samples.

## 3. Four canonical methods

The active IDs are exactly `bytetrack_raw`, `hybrid_bytetrack`,
`realtime_fast`, and `rf_hybrid`. `rf_hybrid v2` is a rejected candidate,
standardized B1 is forensic-only, and no symmetric 2x2 claim is active.

## 4. Prediction and execution authorities

`bytetrack_raw` and `realtime_fast` have current executable prediction
authorities. Historical XMLs remain primary for `hybrid_bytetrack`; its full
accepted lineage is recovered but exact historical runtime is not. `rf_hybrid`
v1 is a frozen development transfer ablation.

## 5. Development population and dependence structure

All methods use the same 13 videos, frames 0-1799, GT hash, Standard V2
contract, 19 HOTA thresholds, IoU 0.50 eligibility, IDSW_STANDARD, and primary
`include_hidden=true` policy. No documented camera/pen/session metadata proves
cross-video independence. Video is therefore the descriptive cluster unit;
session aggregation is not justified and frame-level inference is forbidden.

## 6. Canonical aggregate results

| Method | HOTA | IDF1 | IDSW | FP/FN | Fragments | Wrong-ID | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for method in EXPECTED_METHODS:
        row = by_method[method]
        text += (
            f"| {method} | {row['HOTA']:.6f} | {row['IDF1']:.6f} | "
            f"{row['IDSW_STANDARD']} | {row['FP']}/{row['FN']} | "
            f"{row['fragments']} | {row['wrong_identity_frames']} | "
            f"{row['terminal_identity_episodes']} |\n"
        )
    text += f"""

## 7. Per-video/session consistency

All 52 method-video rows are supplied. Raw counts reproduce canonical global
totals. HOTA and IDF1 are not arithmetic means: HOTA combines per-alpha
sufficient statistics, while IDF1 recomputes from summed identity counts.
Per-session results are not produced because session grouping is unproven.

## 8. Paired comparisons

- C1 realtime_fast minus bytetrack_raw mean per-video HOTA difference:
  {c1_hota["mean_paired_difference"]:.6f}; W/T/L =
  {c1_hota["wins"]}/{c1_hota["ties"]}/{c1_hota["losses"]}.
- C2 hybrid_bytetrack minus realtime_fast mean per-video HOTA difference:
  {c2_hota["mean_paired_difference"]:.6f}; W/T/L =
  {c2_hota["wins"]}/{c2_hota["ties"]}/{c2_hota["losses"]}.
- C3 rf_hybrid minus realtime_fast mean per-video HOTA difference:
  {c3_hota["mean_paired_difference"]:.6f}; W/T/L =
  {c3_hota["wins"]}/{c3_hota["ties"]}/{c3_hota["losses"]}.

Aggregate advantage is never labeled consistent unless its video directions
support that description.

## 9. Uncertainty and influence analysis

Paired video bootstrap uses seed {BOOTSTRAP_SEED} and {BOOTSTRAP_RESAMPLES}
resamples. Percentile ranges are descriptive only. Formal CI status is
`INSUFFICIENT_INDEPENDENT_CLUSTERS_FOR_RELIABLE_CI` because recording-session
independence is unresolved. Leave-one-video-out results identify influential
clusters and replace frame-level significance claims.

## 10. Detection–association trade-offs

Hybrid has stronger AssA/IDF1 and identity severity but lower DetA and larger
FP/FN/fragments than the executable methods. It is therefore not best in every
dimension. rf_hybrid slightly improves detection counts while degrading broad
association quality and identity exposure.

## 11. IDSW=0 and fragmentation analysis

Conclusion: `{hybrid_defense["conclusion"]}`. Zero IDSW is supported jointly by
IDF1=0.991501, AssA=0.911904, only 24 wrong-ID animal-frames, eight recovered
short episodes, zero terminal episodes, and zero persistent swaps. The 425
Standard V2 fragments remain a real detection/localization continuity cost.
The same historical prediction has only six gaps exceeding the supplementary
15-frame gap-tolerant rule, indicating most strict fragments are short gaps,
not persistent owner loss. This does not substitute for human trajectory audit.

## 12. Hidden/visible sensitivity

Conclusion: `{hidden_conclusion}`. Visible-only results are secondary and do
not replace the primary include-Hidden authority. Exclusion tests sensitivity;
it does not validate or correct tracker-derived Hidden labels. The hybrid-minus-
realtime HOTA difference is {hybrid_primary_advantage:.6f} under the primary
policy and {hybrid_visible_advantage:.6f} visible-only, so the development HOTA
advantage is not concentrated in Hidden observations. Realtime wrong-ID exposure
changes from {realtime_primary_wrong:,} to {realtime_visible_wrong:,} animal-
frames ({(realtime_primary_wrong - realtime_visible_wrong) / realtime_primary_wrong:.1%}
lower), but this cannot be interpreted as a causal occlusion-error fraction
because exclusion changes the matching population. Largest absolute per-video
HOTA changes are:
{top_hidden_text}.

## 13. Identity-error and GT audit status

The non-mutating audit pack includes all 24 hybrid wrong-ID animal-frames, all
hybrid episodes, all realtime terminal episodes, every persistent pairwise
swap, long raw-baseline episodes, recovered samples, Hidden-policy references,
and influential videos. `HUMAN_GT_AUDIT_COMPLETED=NO`; every item is explicitly
`NOT_REVIEWED`.

## 14. Hybrid historical-versus-rerun reproducibility

Historical HOTA is 0.900291 and the latest full rerun is
{rerun["hota"]:.6f}; historical IDF1 is 0.991501 and rerun IDF1 is
{rerun["idf1"]:.6f}. Both have IDSW=0 and 24 wrong-ID frames. This establishes
metric-level near parity, not byte-exact prediction reproduction. Historical
XMLs remain primary and runtime provenance remains incomplete.

## 15. Complete-method fairness

C1 is a complete-method comparison; C2 is a complete-method comparison with a
historical-runtime limitation; C3 is a transfer ablation. No result supports a
pure association-core, single-stage causal, detector-controlled, or identical
topology claim.

## 16. Causal-versus-realtime status

`realtime_fast` has zero-delay causal semantics, but the frozen runtime
benchmark protocol has not been executed. The defensible wording is "causal
realtime-oriented method," not "real-time system."

## 17. Baseline adequacy

ByteTrack is an appropriate conventional baseline for a behavior-pipeline-
primary paper. Internal variants are not external baselines. An additional MOT
baseline is recommended—and required for a strong tracking-method-primary
claim—but is not run in this task.

## 18. Negative rf_hybrid transfer result

rf_hybrid reduces IDSW 29 to 18, FP 473 to 412, FN 597 to 536, and fragments
107 to 87. It also decreases HOTA 0.888187 to 0.878281 and IDF1 0.971892 to
0.957881, while increasing wrong-ID frames 11,893 to 14,515 and terminal
episodes 12 to 14. This is a scientifically valid mixed negative result.

## 19. Claims supported on development

The claim matrix contains {strong} strongly supported development claims and
{major} claims supported with major limitations. The strongest statements are
limited explicitly to the frozen development population.

## 20. Claims requiring unseen evaluation

{unseen} matrix claim(s) are explicitly classified `REQUIRES_UNSEEN`; several
other development claims also require the 12-video set before generalization.

## 21. Likely reviewer objections and responses

Thirteen neutral objection responses are supplied. The most serious risks are
development overfitting/no unseen generalization, unresolved session
independence, and historical hybrid runtime incompleteness. None is concealed.

## 22. Limitations

- Development and method iteration share the same 13-video population.
- Recording-session independence cannot be proven from existing metadata.
- Hidden labels are not fully human-validated.
- Exact historical hybrid prediction/runtime reproduction is unavailable.
- The standardized runtime protocol is not executed.
- One conventional external tracker baseline is available.
- Mostly-tracked/partially-tracked/mostly-lost breakdown is unavailable.

## 23. Paper-ready wording

"On the frozen 13-video development set, the historical offline hybrid had the
highest HOTA and strongest identity-continuity diagnostics. The causal
realtime-oriented method improved broadly over the current executable
ByteTrack baseline as a complete pipeline, not as a detector-controlled
association-core ablation. Transfer of selected hybrid mechanisms into the RF
tracklets reduced IDSW but worsened HOTA, IDF1, and wrong-identity exposure."

## 24. Remaining work before final submission

1. Freeze and run the separate 12-video evaluation without method changes.
2. Complete human review of the generated GT/error audit pack.
3. Execute the frozen paired runtime protocol before any real-time claim.
4. Add an external MOT baseline only if tracking becomes a primary paper claim.
5. Report session-cluster inference only if authoritative session metadata is
   recovered.

This document is development evidence and reviewer preparation, not final
external validation. Unsupported claims in the matrix: {unsupported}.
"""
    (
        output_dir / "TRACKING_DEVELOPMENT_EVIDENCE_AND_REVIEWER_DEFENSE_REPORT_20260730.md"
    ).write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def build_final_authority(
    output_dir: Path,
    verified: dict[str, Any],
    claims: list[dict[str, Any]],
    hidden_conclusion: str,
    hybrid_defense: dict[str, Any],
) -> None:
    strong = sum(row["support_level"] == "STRONGLY_SUPPORTED_ON_DEVELOPMENT" for row in claims)
    major = sum(row["support_level"] == "SUPPORTED_WITH_MAJOR_LIMITATION" for row in claims)
    unseen = sum(row["support_level"] == "REQUIRES_UNSEEN" for row in claims)
    unsupported = sum(row["support_level"] == "NOT_SUPPORTED" for row in claims)
    write_json(
        output_dir / "DEVELOPMENT_EVIDENCE_FINAL_DECISION_20260730.json",
        {
            "STARTING_MAIN_SHA": verified["starting_main_sha"],
            "ACTIVE_METHOD_COUNT": 4,
            "ACTIVE_METHODS": list(EXPECTED_METHODS),
            "DEVELOPMENT_VIDEOS": 13,
            "DEVELOPMENT_FRAMES": 23400,
            "PROVEN_INDEPENDENT_CLUSTERS": "UNKNOWN",
            "STATISTICAL_CLUSTER_UNIT": "VIDEO",
            "PER_VIDEO_ANALYSIS_COMPLETED": "YES",
            "PER_SESSION_ANALYSIS_COMPLETED": "NOT_JUSTIFIED",
            "PAIRED_COMPARISON_COMPLETED": "YES",
            "CLUSTER_BOOTSTRAP_COMPLETED": "YES_DESCRIPTIVE_ONLY",
            "LEAVE_ONE_CLUSTER_OUT_COMPLETED": "YES",
            "HIDDEN_SENSITIVITY_COMPLETED": "YES",
            "HIDDEN_SENSITIVITY_CONCLUSION": hidden_conclusion,
            "FRAGMENTATION_AUDIT_COMPLETED": "YES",
            "HYBRID_IDSW_ZERO_CONCLUSION": hybrid_defense["conclusion"],
            "GT_ERROR_AUDIT_PACK_CREATED": "YES",
            "HUMAN_GT_AUDIT_COMPLETED": "NO",
            "CUSTOM_METRIC_SPECIFICATION_STATUS": "MATCHES_FROZEN_IMPLEMENTATION",
            "COMPLETE_METHOD_FAIRNESS_STATUS": "PASS_WITH_EXPLICIT_TOPOLOGY_DIFFERENCES",
            "RUNTIME_CLAIM_STATUS": "REALTIME_ORIENTED_BUT_NOT_YET_BENCHMARKED",
            "BASELINE_ADEQUACY_STATUS": "CURRENT_BASELINE_SUFFICIENT_FOR_UPSTREAM_MODULE_STUDY",
            "HYBRID_EXACT_REPRODUCIBILITY": "NO",
            "HYBRID_METRIC_LEVEL_NEAR_PARITY": "YES",
            "DEVELOPMENT_OVERFITTING_DISCLOSED": "YES",
            "FRAME_LEVEL_INFERENCE_USED": "NO",
            "GENERALIZATION_CLAIM_MADE": "NO",
            "PREDICTION_FILES_CHANGED": 0,
            "GT_FILES_CHANGED": 0,
            "DETECTOR_RUNS": 0,
            "TRACKER_RUNS": 0,
            "EVALUATOR_MATHEMATICS_CHANGED": "NO",
            "POST_RESULT_PARAMETER_TUNING": 0,
            "PER_VIDEO_OVERRIDES": 0,
            "UNSEEN_FILES_ACCESSED": 0,
            "MP4_FILES_CREATED": 0,
            "FINAL_DECISION": "PASS_STRONG_DEVELOPMENT_EVIDENCE_WITH_EXPLICIT_LIMITATIONS",
            "STRONGLY_SUPPORTED_DEVELOPMENT_CLAIMS": strong,
            "CLAIMS_WITH_MAJOR_LIMITATIONS": major,
            "CLAIMS_REQUIRING_UNSEEN": unseen,
            "UNSUPPORTED_CLAIMS": unsupported,
            "MOST_SERIOUS_CURRENT_REVIEWER_RISK": "NO_UNSEEN_GENERALIZATION",
            "READY_TO_CLOSE_TRACKING_DEVELOPMENT_ANALYSIS": "YES",
            "READY_FOR_FUTURE_12_VIDEO_EVALUATION_PREPARATION": "YES",
            "NEXT_AUTHORIZED_ACTION": "FREEZE_12_VIDEO_EVALUATION_CONTRACT_WITHOUT_ACCESSING_DATA",
        },
    )
    files = [
        file_record(path, output_dir, "DEVELOPMENT_EVIDENCE_ARTIFACT")
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "DEVELOPMENT_EVIDENCE_ARTIFACT_MANIFEST_20260730.json"
    ]
    write_json(
        output_dir / "DEVELOPMENT_EVIDENCE_ARTIFACT_MANIFEST_20260730.json",
        {
            "schema_version": "tracking.development_evidence.artifacts.v1",
            "artifact_count": len(files),
            "artifacts": files,
            "canonical_sha256": collection_hash(files),
            "recursive_mp4_count": 0,
        },
    )


def execute(source_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise EvidenceError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    verified = verify_authorities(source_root, REPO)
    population_payload = read_json(source_root / LOCKED_POPULATION)
    population = population_payload["videos"]
    if len(population) != EXPECTED_VIDEOS:
        raise EvidenceError("Locked development population is not 13 videos")
    canonical = canonical_rows(verified["canonical"])
    per_video = normalize_per_video(source_root, population)
    count_check = verify_global_counts(per_video, canonical)
    build_input_authority(source_root, output_dir, verified, population)
    build_session_map(output_dir, population)
    build_per_video(output_dir, per_video)
    write_json(output_dir / "DEVELOPMENT_PER_VIDEO_AGGREGATION_CHECK_20260730.json", count_check)
    paired, summaries, loco = build_paired(output_dir, per_video)
    build_bootstrap(output_dir, paired)
    hidden_rows, _visible = build_hidden_sensitivity(
        source_root,
        output_dir,
        population,
        canonical,
        per_video,
    )
    hidden_decision = read_json(
        output_dir / "DEVELOPMENT_HIDDEN_SENSITIVITY_DECISION_20260730.json"
    )["conclusion"]
    hybrid_defense = build_fragmentation(
        source_root,
        output_dir,
        population,
        canonical,
        per_video,
    )
    build_audit_pack(source_root, output_dir, population, summaries)
    write_metric_specification(output_dir)
    build_fairness(output_dir, canonical)
    build_baseline_runtime(output_dir)
    claims = build_claims(output_dir, hidden_decision)
    objections = build_objections(output_dir, hidden_decision, hybrid_defense)
    build_reproducibility_table(output_dir, canonical, verified["rerun"])
    build_paper_tables(
        output_dir,
        canonical,
        paired,
        hidden_rows,
        claims,
        per_video,
        loco,
    )
    build_report(
        output_dir,
        canonical,
        summaries,
        hidden_decision,
        hybrid_defense,
        claims,
        objections,
        verified["rerun"],
    )
    build_final_authority(
        output_dir,
        verified,
        claims,
        hidden_decision,
        hybrid_defense,
    )
    if list(output_dir.rglob("*.mp4")):
        raise EvidenceError("Unauthorized MP4 artifact created")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs/tracking/development_evidence_defense",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute(args.source_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
