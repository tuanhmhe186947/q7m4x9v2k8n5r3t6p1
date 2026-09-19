"""Freeze and evaluate surviving historical H5b/H4 predictions under Standard V2."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_b0_b1_r0_standard_v2 as baseline  # noqa: E402

from pig_behavior.evaluation.tracking.contracts import (  # noqa: E402
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    IDENTITY_EPISODE_CONTRACT_ID,
    IDSW_POLICY,
    MATCHING_CONTRACT_ID,
    REFERENCE_PARITY_PASS,
    SEQUENCE_BOUNDARY_POLICY,
    resolve_evaluator_code_sha,
)

DATE = "20260728"
STARTING_MAIN_SHA = "3035db0cec10ecbd8ccf49bb38c43ce2f8ca3823"
HISTORICAL_RUN_ID = "20260719_h5b_h4_full13_combined_v2"
HISTORICAL_SOURCE_SHA = "31d360ba96b4065ce5125c0d88765531cc5898ae"
HISTORICAL_PROFILE = "hybrid_bytetrack_best"
HISTORICAL_DETECTOR_CONFIDENCE = 0.20
HISTORICAL_MAX_RAW_DETECTIONS = 64
CURRENT_B1_DETECTOR_CONFIDENCE = 0.25
CURRENT_B1_MAX_RAW_DETECTIONS = 32
DETECTOR_WEIGHTS_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
EXPECTED_ARTIFACT_COUNT = 23
EXPECTED_PREDICTION_COUNT = 13
HISTORICAL_ARM = "HISTORICAL_H5B_H4"

OUTPUT_NAME_MAP = {
    baseline.REQUIRED_REPEAT_FILES[0]: (
        "HISTORICAL_H5B_H4_STANDARD_V2_AGGREGATE_METRICS.csv"
    ),
    baseline.REQUIRED_REPEAT_FILES[1]: (
        "HISTORICAL_H5B_H4_STANDARD_V2_PER_VIDEO_METRICS.csv"
    ),
    baseline.REQUIRED_REPEAT_FILES[2]: (
        "HISTORICAL_H5B_H4_STANDARD_V2_PER_ALPHA_METRICS.csv"
    ),
    baseline.REQUIRED_REPEAT_FILES[3]: (
        "HISTORICAL_H5B_H4_IDENTITY_ERROR_EPISODES.csv"
    ),
    baseline.REQUIRED_REPEAT_FILES[4]: (
        "HISTORICAL_H5B_H4_PERSISTENT_PAIRWISE_SWAPS.csv"
    ),
    baseline.REQUIRED_REPEAT_FILES[5]: (
        "HISTORICAL_H5B_H4_EXPOSURE_METRICS.csv"
    ),
}

COMPARISON_METRICS = {
    "hota": (True, "ratio"),
    "deta": (True, "ratio"),
    "assa": (True, "ratio"),
    "loca": (True, "ratio"),
    "idf1": (True, "ratio"),
    "id_precision": (True, "ratio"),
    "id_recall": (True, "ratio"),
    "idsw_standard": (False, "count"),
    "fp": (False, "count"),
    "fn": (False, "count"),
    "fragments": (False, "count"),
    "wrong_id_matched_frames": (False, "frames"),
    "wrong_id_matched_seconds": (False, "seconds"),
    "identity_error_episode_count": (False, "count"),
    "recovered_identity_error_episode_count": (False, "count"),
    "terminal_identity_error_episode_count": (False, "count"),
    "persistent_pairwise_identity_swap_count": (False, "count"),
}


class CorrectiveEvaluationError(RuntimeError):
    """Fail closed when historical or current authority evidence differs."""


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _historical_manifest_path(historical_repo: Path) -> Path:
    return (
        historical_repo
        / "outputs"
        / "eval"
        / "tracking_hybrid_residual"
        / HISTORICAL_RUN_ID
        / "artifact_manifest.json"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = baseline.load_json(path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise CorrectiveEvaluationError("Historical artifact list is missing")
    if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise CorrectiveEvaluationError("Historical artifact count mismatch")
    prediction_count = sum(row.get("role") == "prediction_xml" for row in artifacts)
    if prediction_count != EXPECTED_PREDICTION_COUNT:
        raise CorrectiveEvaluationError("Historical prediction count mismatch")
    return payload


def _xml_record(path: Path, video_key: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    frames: list[int] = []
    object_count = 0
    for box in root.findall("./track/box"):
        if str(box.attrib.get("outside", "0")).strip() in {"1", "true", "True"}:
            continue
        frames.append(int(float(box.attrib["frame"])))
        object_count += 1
    if not frames:
        raise CorrectiveEvaluationError(f"No active prediction rows: {video_key}")
    return {
        "video_key": video_key,
        "file_size": path.stat().st_size,
        "sha256": baseline.sha256_file(path),
        "object_count": object_count,
        "frame_start": min(frames),
        "frame_end": max(frames),
    }


def _copy_historical_artifacts(
    historical_repo: Path,
    frozen_root: Path,
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    if frozen_root.exists():
        raise CorrectiveEvaluationError(f"Refusing overwrite: {frozen_root}")
    manifest_path = _historical_manifest_path(historical_repo)
    manifest = _load_manifest(manifest_path)
    frozen_root.mkdir(parents=True)
    population = {row["video_key"]: row for row in videos}
    copied: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for artifact in manifest["artifacts"]:
        source = Path(str(artifact["path"]))
        if not source.is_file():
            raise CorrectiveEvaluationError(f"Missing historical artifact: {source}")
        source_sha = baseline.sha256_file(source)
        if source_sha != artifact["sha256"]:
            raise CorrectiveEvaluationError(f"Historical hash mismatch: {source}")
        if int(source.stat().st_size) != int(artifact["size_bytes"]):
            raise CorrectiveEvaluationError(f"Historical size mismatch: {source}")
        role = str(artifact["role"])
        if role == "prediction_xml":
            video_key = source.parent.name
            if video_key not in population:
                raise CorrectiveEvaluationError(
                    f"Historical video is outside locked population: {video_key}"
                )
            destination = frozen_root / "predictions" / f"{video_key}.xml"
        else:
            video_key = None
            destination = frozen_root / "legacy_evaluation" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination_sha = baseline.sha256_file(destination)
        if source_sha != destination_sha:
            raise CorrectiveEvaluationError(f"Copy parity failed: {source}")
        record = {
            "artifact_role": role,
            "original_path": str(source),
            "copied_path": str(destination),
            "file_size": destination.stat().st_size,
            "source_sha256": source_sha,
            "destination_sha256": destination_sha,
            "byte_parity": "PASS",
        }
        if video_key is not None:
            xml_record = _xml_record(destination, video_key)
            video = population[video_key]
            xml_record.update(
                {
                    "original_path": str(source),
                    "copied_path": str(destination),
                    "source_video_sha256": video["source_video_sha256"],
                    "gt_sha256": video["gt_sha256"],
                    "artifact_role": role,
                }
            )
            prediction_records.append(xml_record)
            record["video_key"] = video_key
        copied.append(record)
    copied_manifest = frozen_root / "source_artifact_manifest.json"
    shutil.copy2(manifest_path, copied_manifest)
    if baseline.sha256_file(copied_manifest) != baseline.sha256_file(manifest_path):
        raise CorrectiveEvaluationError("Source manifest copy parity failed")
    prediction_records.sort(key=lambda row: row["video_key"])
    copied.sort(key=lambda row: (row["artifact_role"], row["copied_path"]))
    prediction_artifact_sha = baseline.canonical_hash(
        [
            {
                "video_key": row["video_key"],
                "sha256": row["sha256"],
                "file_size": row["file_size"],
            }
            for row in prediction_records
        ]
    )
    authority_manifest = {
        "schema_version": "tracking.historical_h5b_h4.prediction_manifest.v1",
        "date": DATE,
        "status": "ESTABLISHED",
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "profile": HISTORICAL_PROFILE,
        "retention_class": (
            "NON_DISPOSABLE_FROZEN_HISTORICAL_PREDICTION_AUTHORITY"
        ),
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "historical_manifest_sha256": baseline.sha256_file(manifest_path),
        "historical_manifest_artifact_count": len(manifest["artifacts"]),
        "prediction_xml_count": len(prediction_records),
        "prediction_artifact_sha256": prediction_artifact_sha,
        "historical_effective_detector_configuration": {
            "confidence": HISTORICAL_DETECTOR_CONFIDENCE,
            "max_raw_detections": HISTORICAL_MAX_RAW_DETECTIONS,
            "weights_sha256": DETECTOR_WEIGHTS_SHA256,
        },
        "tracker_repair_logic_match_current_b1": True,
        "effective_detector_pipeline_match_current_b1": False,
        "population": prediction_records,
        "artifacts": copied,
        "legacy_evaluator_contract": "TRACKING_EVALUATOR_LEGACY_V1",
    }
    conservation = {
        "schema_version": "tracking.historical_h5b_h4.copy_conservation.v1",
        "date": DATE,
        "status": "PASS",
        "source_artifact_count": len(copied),
        "destination_artifact_count": len(copied),
        "byte_parity_passed": len(copied),
        "byte_parity_failed": 0,
        "source_manifest_copy_parity": "PASS",
    }
    baseline.write_json(
        frozen_root
        / "HISTORICAL_H5B_H4_PREDICTION_ARTIFACT_MANIFEST_20260728.json",
        authority_manifest,
    )
    baseline.write_json(
        frozen_root / "HISTORICAL_H5B_H4_COPY_CONSERVATION_20260728.json",
        conservation,
    )
    (frozen_root / "FROZEN_HISTORICAL_AUTHORITY_DO_NOT_DELETE.txt").write_text(
        "NON_DISPOSABLE_FROZEN_HISTORICAL_PREDICTION_AUTHORITY\n"
        "Deletion requires explicit authority retirement.\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "manifest": authority_manifest,
        "conservation": conservation,
        "source_hashes": {row["original_path"]: row["source_sha256"] for row in copied},
    }


def _historical_arm(frozen_root: Path, artifact_sha: str) -> baseline.ArmSpec:
    effective_config = {
        "profile": HISTORICAL_PROFILE,
        "detector_confidence": HISTORICAL_DETECTOR_CONFIDENCE,
        "max_raw_detections": HISTORICAL_MAX_RAW_DETECTIONS,
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "tracker_repair_logic_match_current_b1": True,
    }
    return baseline.ArmSpec(
        arm=HISTORICAL_ARM,
        profile=HISTORICAL_PROFILE,
        prediction_root=frozen_root / "predictions",
        authority_path=frozen_root
        / "HISTORICAL_H5B_H4_PREDICTION_ARTIFACT_MANIFEST_20260728.json",
        artifact_sha256=artifact_sha,
        config_sha256=baseline.canonical_hash(effective_config),
        detector_cadence="EVERY_FRAME",
        detector_authority_sha256=baseline.canonical_hash(
            {
                "confidence": HISTORICAL_DETECTOR_CONFIDENCE,
                "max_raw_detections": HISTORICAL_MAX_RAW_DETECTIONS,
                "weights_sha256": DETECTOR_WEIGHTS_SHA256,
            }
        ),
    )


def _historical_videos(
    current_videos: list[dict[str, Any]], frozen_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for source in current_videos:
        row = dict(source)
        prediction = frozen_root / "predictions" / f"{row['video_key']}.xml"
        row["prediction_paths"] = {HISTORICAL_ARM: str(prediction)}
        rows.append(row)
    return rows


def _copy_metric_outputs(pass_root: Path, result_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source_name, destination_name in OUTPUT_NAME_MAP.items():
        destination = result_root / destination_name
        shutil.copy2(pass_root / source_name, destination)
        hashes[destination_name] = baseline.sha256_file(destination)
    return hashes


def _comparison_table(
    historical: pd.Series, current_aggregate: pd.DataFrame
) -> pd.DataFrame:
    by_arm = {
        str(row["arm"]): row for _, row in current_aggregate.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for metric, (higher_is_better, unit) in COMPARISON_METRICS.items():
        historical_value = float(historical[metric])
        b1 = float(by_arm["B1"][metric])
        r0 = float(by_arm["R0"][metric])
        b0 = float(by_arm["B0"][metric])
        rows.append(
            {
                "metric": metric,
                "higher_is_better": higher_is_better,
                "historical_h5b_h4": historical_value,
                "current_B1": b1,
                "current_R0": r0,
                "current_B0": b0,
                "historical_minus_current_B1": historical_value - b1,
                "historical_minus_R0": historical_value - r0,
                "historical_minus_B0": historical_value - b0,
                "unit": unit,
                "interpretation": (
                    "whole-pipeline prediction-level comparison; not a pure "
                    "association-core effect"
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_vs_r0(historical: pd.Series, r0: pd.Series) -> str:
    historical_better = (
        historical["hota"] > r0["hota"]
        and historical["idf1"] > r0["idf1"]
        and historical["wrong_id_matched_frames"]
        <= r0["wrong_id_matched_frames"]
        and historical["terminal_identity_error_episode_count"]
        <= r0["terminal_identity_error_episode_count"]
        and historical["persistent_pairwise_identity_swap_count"]
        <= r0["persistent_pairwise_identity_swap_count"]
    )
    r0_better = (
        r0["hota"] > historical["hota"]
        and r0["idf1"] > historical["idf1"]
        and r0["wrong_id_matched_frames"]
        <= historical["wrong_id_matched_frames"]
        and r0["terminal_identity_error_episode_count"]
        <= historical["terminal_identity_error_episode_count"]
        and r0["persistent_pairwise_identity_swap_count"]
        <= historical["persistent_pairwise_identity_swap_count"]
    )
    if historical_better:
        return "HISTORICAL_HYBRID_BROADLY_BETTER_THAN_R0"
    if r0_better:
        return "R0_BROADLY_BETTER_THAN_HISTORICAL_HYBRID"
    equal_fields = (
        "hota",
        "idf1",
        "wrong_id_matched_frames",
        "terminal_identity_error_episode_count",
        "persistent_pairwise_identity_swap_count",
    )
    if all(historical[field] == r0[field] for field in equal_fields):
        return "NO_OBSERVED_DIFFERENCE"
    return "HISTORICAL_HYBRID_MIXED_VS_R0"


def _impact_decision(classification: str) -> tuple[str, str]:
    if classification == "HISTORICAL_HYBRID_BROADLY_BETTER_THAN_R0":
        return (
            "SUSPEND_CURRENT_UNSEEN_FREEZE_PENDING_HISTORICAL_METHOD_REPRODUCTION",
            "SUSPENDED_PENDING_REPRODUCTION",
        )
    if classification in {
        "R0_BROADLY_BETTER_THAN_HISTORICAL_HYBRID",
        "HISTORICAL_HYBRID_MIXED_VS_R0",
        "NO_OBSERVED_DIFFERENCE",
    }:
        return ("REAFFIRM_CURRENT_UNSEEN_METHOD_FREEZE", "REAFFIRMED")
    return ("INCONCLUSIVE", "INCONCLUSIVE")


def _reconciliation_markdown(historical: pd.Series) -> str:
    legacy_idsw = 0
    standard_idsw = int(historical["idsw_standard"])
    return (
        "# Historical H5b/H4 Legacy–Standard-V2 Reconciliation\n\n"
        f"- Legacy evaluator: `TRACKING_EVALUATOR_LEGACY_V1`\n"
        f"- Legacy HOTA: `0.9835062270290739`\n"
        f"- Standard-V2 HOTA: `{historical['hota']}`\n"
        f"- Legacy IDF1: `0.9914903846153846`\n"
        f"- Standard-V2 IDF1: `{historical['idf1']}`\n"
        f"- Legacy IDSW: `{legacy_idsw}`\n"
        f"- Standard-V2 IDSW: `{standard_idsw}`\n\n"
        "The prediction bytes are unchanged. The corrected values differ only "
        "because Standard V2 uses pre-assignment eligibility, video-isolated "
        "sequences, Hidden-inclusive matching, the frozen 19-alpha HOTA set, "
        "and standard last-match IDSW semantics. Therefore this is evaluator "
        "reconciliation, not a tracker regression.\n\n"
        f"Legacy IDSW=0 survives Standard V2: "
        f"`{'YES' if standard_idsw == 0 else 'NO'}`.\n"
    )


def _source_integrity(
    historical_repo: Path,
    head_before: str,
    status_before: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    changed = []
    for raw_path, expected in source_hashes.items():
        actual = baseline.sha256_file(Path(raw_path))
        if actual != expected:
            changed.append(raw_path)
    head_after = _git(historical_repo, "rev-parse", "HEAD")
    status_after = _git(historical_repo, "status", "--porcelain=v1", "-uall")
    return {
        "head_before": head_before,
        "head_after": head_after,
        "worktree_head_changed": head_after != head_before,
        "status_before": status_before,
        "status_after": status_after,
        "staged_unstaged_untracked_state_changed": status_after != status_before,
        "tracked_source_artifacts_changed": changed,
    }


def freeze(
    source_repo: Path,
    worktree_repo: Path,
    historical_repo: Path,
    frozen_root: Path,
    result_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    if result_root.exists():
        raise CorrectiveEvaluationError(f"Refusing overwrite: {result_root}")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise CorrectiveEvaluationError("PYTHONDONTWRITEBYTECODE=1 is required")
    current_head = _git(worktree_repo, "rev-parse", "HEAD")
    if current_head == STARTING_MAIN_SHA:
        raise CorrectiveEvaluationError(
            "Commit the frozen plan and orchestration before evaluation"
        )
    if not _git(
        worktree_repo,
        "merge-base",
        "--is-ancestor",
        STARTING_MAIN_SHA,
        current_head,
    ) == "":
        raise CorrectiveEvaluationError("Starting main is not an ancestor")
    historical_head = _git(historical_repo, "rev-parse", "HEAD")
    historical_status = _git(
        historical_repo, "status", "--porcelain=v1", "-uall"
    )
    current_videos, _, _, _ = baseline.preflight(source_repo, worktree_repo)
    copy_state = _copy_historical_artifacts(
        historical_repo, frozen_root, current_videos
    )
    historical_videos = _historical_videos(current_videos, frozen_root)
    arm = _historical_arm(
        frozen_root, copy_state["manifest"]["prediction_artifact_sha256"]
    )
    result_root.mkdir(parents=True)
    evaluator_sha = resolve_evaluator_code_sha()
    if evaluator_sha != current_head:
        raise CorrectiveEvaluationError("Evaluator code SHA is not current HEAD")
    pass1 = baseline.evaluate_pass(
        result_root / "pass1",
        (arm,),
        historical_videos,
        evaluator_code_sha=evaluator_sha,
        reverse_inputs=False,
    )
    pass2 = baseline.evaluate_pass(
        result_root / "pass2",
        (arm,),
        historical_videos,
        evaluator_code_sha=evaluator_sha,
        reverse_inputs=True,
    )
    determinism = baseline.compare_passes(pass1, pass2)
    conservation = pass1["conservation"]
    if determinism["reevaluation_repeatability"] != "PASS":
        raise CorrectiveEvaluationError("Deterministic repeat failed")
    required_conservation = (
        conservation["tp_fp_fn_conservation"] == "PASS"
        and conservation["wrong_id_row_conservation"] == "PASS"
        and conservation["identity_episode_double_count"] == 0
        and conservation["pairwise_swap_double_count"] == 0
        and conservation["multi_video_boundary_status"] == "PASS"
    )
    if not required_conservation:
        raise CorrectiveEvaluationError("Metric conservation failed")
    metric_hashes = _copy_metric_outputs(result_root / "pass1", result_root)
    historical_aggregate = pass1["aggregate_dataframe"].iloc[0]
    development_authority_path = (
        worktree_repo
        / "docs"
        / "tracking"
        / "development_2x2_standard_v2"
        / "DEVELOPMENT_2X2_STANDARD_V2_AUTHORITY_20260728.json"
    )
    development_authority = baseline.load_json(development_authority_path)
    baseline_authority = development_authority["baseline_metric_authority"]
    aggregate_path = (
        Path(baseline_authority["result_root"])
        / "B0_B1_R0_STANDARD_V2_AGGREGATE_METRICS.csv"
    )
    if baseline.sha256_file(aggregate_path) != baseline_authority[
        "aggregate_sha256"
    ]:
        raise CorrectiveEvaluationError("Current aggregate authority hash failed")
    current_aggregate = pd.read_csv(aggregate_path)
    comparison = _comparison_table(historical_aggregate, current_aggregate)
    comparison_name = "HISTORICAL_H5B_H4_VS_CURRENT_AUTHORITIES_20260728.csv"
    baseline.write_csv(comparison, result_root / comparison_name)
    comparison_hash = baseline.sha256_file(result_root / comparison_name)
    r0 = current_aggregate.loc[current_aggregate["arm"] == "R0"].iloc[0]
    classification = classify_vs_r0(historical_aggregate, r0)
    impact, freeze_status = _impact_decision(classification)
    legacy_zero_survives = int(historical_aggregate["idsw_standard"]) == 0
    reconciliation_name = (
        "HISTORICAL_H5B_H4_LEGACY_STANDARD_V2_RECONCILIATION_20260728.md"
    )
    reconciliation = _reconciliation_markdown(historical_aggregate)
    (result_root / reconciliation_name).write_text(
        reconciliation, encoding="utf-8", newline="\n"
    )
    source_integrity = _source_integrity(
        historical_repo,
        historical_head,
        historical_status,
        copy_state["source_hashes"],
    )
    if (
        source_integrity["worktree_head_changed"]
        or source_integrity["staged_unstaged_untracked_state_changed"]
        or source_integrity["tracked_source_artifacts_changed"]
    ):
        raise CorrectiveEvaluationError("Historical source integrity changed")
    post_copy_hashes = {
        row["copied_path"]: baseline.sha256_file(Path(row["copied_path"]))
        for row in copy_state["manifest"]["artifacts"]
    }
    if any(
        row["destination_sha256"] != post_copy_hashes[row["copied_path"]]
        for row in copy_state["manifest"]["artifacts"]
    ):
        raise CorrectiveEvaluationError("Frozen prediction bytes changed")
    metric_config_sha = str(historical_aggregate["metric_config_sha256"])
    run_manifest = {
        "schema_version": "tracking.historical_h5b_h4.standard_v2.run.v1",
        "date": DATE,
        "status": "PASS",
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "evaluation_code_sha": evaluator_sha,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "matching_contract_id": MATCHING_CONTRACT_ID,
        "primary_include_hidden": True,
        "hota_alpha_set": list(HOTA_ALPHAS),
        "idsw_policy": IDSW_POLICY,
        "sequence_boundary": SEQUENCE_BOUNDARY_POLICY,
        "reference_parity_status": REFERENCE_PARITY_PASS,
        "metric_config_sha256": metric_config_sha,
        "prediction_artifact_sha256": arm.artifact_sha256,
        "result_root": str(result_root),
        "frozen_prediction_root": str(frozen_root),
        "complete_evaluation_passes": 2,
        "metric_hashes": metric_hashes,
        "comparison_hash": comparison_hash,
        "determinism": determinism,
        "conservation": conservation,
        "execution_counts": {
            "tracker_executions": 0,
            "detector_inference_calls": 0,
            "prediction_reconstructions": 0,
            "prediction_regenerations": 0,
            "unseen_files_accessed": 0,
            "run_root_mp4_count": 0,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    baseline.write_json(
        result_root / "HISTORICAL_H5B_H4_STANDARD_V2_RUN_MANIFEST.json",
        run_manifest,
    )
    baseline.write_json(
        result_root / "HISTORICAL_H5B_H4_STANDARD_V2_DETERMINISM.json",
        determinism,
    )
    baseline.write_json(
        result_root / "HISTORICAL_H5B_H4_STANDARD_V2_CONSERVATION.json",
        conservation,
    )
    docs_root.mkdir(parents=True, exist_ok=True)
    prediction_authority = {
        **copy_state["manifest"],
        "frozen_root": str(frozen_root),
        "copy_conservation": copy_state["conservation"],
        "source_integrity": source_integrity,
    }
    standard_authority = {
        "schema_version": "tracking.historical_h5b_h4.standard_v2.authority.v1",
        "date": DATE,
        "status": "ESTABLISHED",
        "prediction_artifact_sha256": arm.artifact_sha256,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "metric_config_sha256": metric_config_sha,
        "aggregate_metrics": historical_aggregate.to_dict(),
        "metric_hashes": metric_hashes,
        "comparison_table_sha256": comparison_hash,
        "determinism": determinism,
        "conservation": conservation,
        "common_population": {
            "video": "PASS",
            "frame": "PASS",
            "source_video": "PASS",
            "gt": "PASS",
            "hidden_policy": "PASS",
            "sequence_boundary": "PASS",
            "evaluator": "PASS",
        },
        "historical_vs_r0_classification": classification,
        "unseen_freeze_impact": impact,
        "scientific_limitations": [
            "Historical and current B1 differ in detector evidence semantics.",
            "Aggregate prediction differences do not establish causal attribution.",
            "Historical prediction bytes do not constitute executable authority.",
            "No unseen access or promotion is authorized.",
        ],
    }
    impact_document = {
        "schema_version": "tracking.historical_h5b_h4.unseen_impact.v1",
        "date": DATE,
        "decision": impact,
        "current_unseen_method_freeze_status": freeze_status,
        "historical_vs_r0_classification": classification,
        "historical_unseen_execution_authorized": False,
        "unseen_data_accessed": False,
        "required_future_work_if_historical_wins": [
            "Formalize an executable historical-best profile.",
            "Reproduce detector settings confidence=0.20 and max_raw_detections=64.",
            "Freeze the complete effective configuration.",
            "Prove deterministic development reproduction.",
            "Issue a separate superseding unseen-method freeze decision.",
        ],
    }
    corrective_decision = {
        "schema_version": "tracking.historical_h5b_h4.corrective_decision.v1",
        "date": DATE,
        "decision": "PASS_HISTORICAL_H5B_H4_STANDARD_V2_AUTHORITY_ESTABLISHED",
        "historical_standard_v2_authority": "ESTABLISHED",
        "historical_vs_r0_classification": classification,
        "unseen_freeze_impact": impact,
        "legacy_idsw_zero_survives_standard_v2": legacy_zero_survives,
        "tracker_repair_logic_match": True,
        "effective_detector_pipeline_match": False,
        "ready_for_unseen_data_authority_freeze": impact
        == "REAFFIRM_CURRENT_UNSEEN_METHOD_FREEZE",
        "ready_for_historical_method_reproduction": impact
        == (
            "SUSPEND_CURRENT_UNSEEN_FREEZE_PENDING_"
            "HISTORICAL_METHOD_REPRODUCTION"
        ),
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
        "execution_counts": run_manifest["execution_counts"],
    }
    baseline.write_json(
        docs_root / "HISTORICAL_H5B_H4_PREDICTION_AUTHORITY_20260728.json",
        prediction_authority,
    )
    baseline.write_json(
        docs_root / "HISTORICAL_H5B_H4_STANDARD_V2_AUTHORITY_20260728.json",
        standard_authority,
    )
    baseline.write_json(
        docs_root / "HISTORICAL_H5B_H4_CORRECTIVE_DECISION_20260728.json",
        corrective_decision,
    )
    baseline.write_json(
        docs_root
        / "HISTORICAL_H5B_H4_UNSEEN_FREEZE_IMPACT_DECISION_20260728.json",
        impact_document,
    )
    (docs_root / reconciliation_name).write_text(
        reconciliation, encoding="utf-8", newline="\n"
    )
    marker = result_root / "FROZEN_SCIENTIFIC_METRIC_AUTHORITY_DO_NOT_DELETE.txt"
    marker.write_text(
        "NON_DISPOSABLE_FROZEN_STANDARD_V2_METRIC_AUTHORITY\n"
        "Deletion requires explicit authority retirement.\n",
        encoding="utf-8",
        newline="\n",
    )
    if any(path.suffix.lower() == ".mp4" for path in result_root.rglob("*")):
        raise CorrectiveEvaluationError("Metric root contains MP4")
    for root in (frozen_root, result_root):
        for path in root.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    return {
        "corrective_decision": corrective_decision,
        "historical_aggregate": historical_aggregate.to_dict(),
        "comparison": comparison.to_dict(orient="records"),
        "source_integrity": source_integrity,
        "frozen_root": str(frozen_root),
        "result_root": str(result_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--historical-repo", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "tracking"
            / "historical_h5b_h4_standard_v2"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze(
        args.source_repo.resolve(),
        args.worktree_repo.resolve(),
        args.historical_repo.resolve(),
        args.frozen_root.resolve(),
        args.result_root.resolve(),
        args.docs_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
