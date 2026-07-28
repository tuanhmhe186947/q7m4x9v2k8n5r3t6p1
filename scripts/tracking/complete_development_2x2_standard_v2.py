"""Evaluate frozen R1 and freeze the development tracking 2x2 authority."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_b0_b1_r0_standard_v2 as baseline  # noqa: E402

from pig_behavior.evaluation.tracking.contracts import (  # noqa: E402
    EVALUATOR_CONTRACT_ID,
    IDENTITY_EPISODE_CONTRACT_ID,
    resolve_evaluator_code_sha,
)
from pig_behavior.evaluation.tracking.cvat_io import (  # noqa: E402
    TrackingObject,
    parse_cvat_video_xml,
)
from pig_behavior.evaluation.tracking.matching_standard_v2 import (  # noqa: E402
    match_frame_eligible,
)

DATE = "20260728"
STARTING_MAIN_SHA = "4049aac2bdedbc6365eb5261aba8fbbabc006256"
R1_ARTIFACT_SHA256 = (
    "40052f992871d50984fc4c0c839c4933b772bca2bfcaaaacafcde40d0e8a1800"
)
R1_REPAIR_SEMANTIC_SHA256 = (
    "e078b5b165dda82dee5b61e9465dc9844446e4cb576a02858c4ed7369828d758"
)
BASELINE_RESULT_RELATIVE = Path(
    "outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1"
)
R1_FILE_MAP = {
    "B0_B1_R0_STANDARD_V2_AGGREGATE_METRICS.csv": (
        "R1_STANDARD_V2_AGGREGATE_METRICS.csv"
    ),
    "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv": (
        "R1_STANDARD_V2_PER_VIDEO_METRICS.csv"
    ),
    "B0_B1_R0_STANDARD_V2_PER_ALPHA_METRICS.csv": (
        "R1_STANDARD_V2_PER_ALPHA_METRICS.csv"
    ),
    "B0_B1_R0_IDENTITY_ERROR_EPISODES.csv": (
        "R1_IDENTITY_ERROR_EPISODES.csv"
    ),
    "B0_B1_R0_PERSISTENT_PAIRWISE_SWAPS.csv": (
        "R1_PERSISTENT_PAIRWISE_SWAPS.csv"
    ),
    "B0_B1_R0_EXPOSURE_NORMALIZED_METRICS.csv": (
        "R1_EXPOSURE_NORMALIZED_METRICS.csv"
    ),
    "B0_B1_R0_ALL_PAIRWISE_SWAP_EVENTS.csv": (
        "R1_ALL_PAIRWISE_SWAP_EVENTS.csv"
    ),
    "B0_B1_R0_IDENTITY_AUTHORITIES.csv": "R1_IDENTITY_AUTHORITIES.csv",
    "B0_B1_R0_IDENTITY_AMBIGUITIES.csv": "R1_IDENTITY_AMBIGUITIES.csv",
    "B0_B1_R0_STANDARD_V2_VIDEO_DESCRIPTIVE_SUMMARY.csv": (
        "R1_STANDARD_V2_VIDEO_DESCRIPTIVE_SUMMARY.csv"
    ),
}
REPEAT_FILES = tuple(R1_FILE_MAP[name] for name in baseline.REQUIRED_REPEAT_FILES)
METRICS = (
    ("hota", "co_primary", True, "ratio"),
    ("idf1", "co_primary", True, "ratio"),
    ("wrong_id_matched_frames", "primary_identity_severity", False, "frames"),
    ("wrong_id_matched_seconds", "primary_identity_severity", False, "seconds"),
    (
        "terminal_identity_error_episode_count",
        "primary_identity_severity",
        False,
        "episodes",
    ),
    (
        "persistent_pairwise_identity_swap_count",
        "primary_identity_severity",
        False,
        "events",
    ),
    ("deta", "supporting", True, "ratio"),
    ("assa", "supporting", True, "ratio"),
    ("loca", "supporting", True, "ratio"),
    ("id_precision", "supporting", True, "ratio"),
    ("id_recall", "supporting", True, "ratio"),
    ("idsw_standard", "supporting", False, "switches"),
    ("fp", "supporting", False, "detections"),
    ("fn", "supporting", False, "detections"),
    ("fragments", "supporting", False, "fragments"),
    ("identity_error_episode_count", "supporting", False, "episodes"),
    ("recovered_identity_error_episode_count", "supporting", False, "episodes"),
)


def r1_spec(source_repo: Path, worktree_repo: Path) -> baseline.ArmSpec:
    """Return the frozen R1 authority binding."""

    root = (
        source_repo
        / "outputs"
        / "tracking"
        / "frozen_predictions_standard_v2_20260728_retry1"
        / "R1_rf_hybrid_offline"
    )
    return baseline.ArmSpec(
        arm="R1",
        profile="rf_hybrid_offline",
        prediction_root=root / "predictions",
        authority_path=(
            worktree_repo
            / "docs"
            / "tracking"
            / "r1_prediction_authority"
            / "R1_RF_HYBRID_OFFLINE_PREDICTION_AUTHORITY_20260728.json"
        ),
        artifact_sha256=R1_ARTIFACT_SHA256,
        config_sha256=R1_REPAIR_SEMANTIC_SHA256,
        detector_cadence="EVERY_2_FRAMES",
        detector_authority_sha256=(
            "795df7732393e4e258a82db58e29101b068cf8ac3583acf7702e0afdaeec6e7a"
        ),
    )


def validate_baseline_metric_authority(
    source_repo: Path,
    worktree_repo: Path,
) -> dict[str, Any]:
    """Verify all reused B0/B1/R0 metric tables against frozen authorities."""

    result_root = source_repo / BASELINE_RESULT_RELATIVE
    docs = (
        worktree_repo / "docs" / "tracking" / "b0_b1_r0_standard_v2"
    )
    authority_names = {
        "B0": "B0_BYTETRACK_RAW_STANDARD_V2_AUTHORITY_20260728.json",
        "B1": "B1_HYBRID_BYTETRACK_STANDARD_V2_AUTHORITY_20260728.json",
        "R0": "R0_REALTIME_FAST_STANDARD_V2_AUTHORITY_20260728.json",
    }
    field_files = {
        "per_video_metric_table_sha256": (
            "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv"
        ),
        "per_alpha_table_sha256": (
            "B0_B1_R0_STANDARD_V2_PER_ALPHA_METRICS.csv"
        ),
        "identity_episode_table_sha256": (
            "B0_B1_R0_IDENTITY_ERROR_EPISODES.csv"
        ),
        "pairwise_swap_table_sha256": (
            "B0_B1_R0_PERSISTENT_PAIRWISE_SWAPS.csv"
        ),
        "exposure_metric_table_sha256": (
            "B0_B1_R0_EXPOSURE_NORMALIZED_METRICS.csv"
        ),
    }
    records: dict[str, Any] = {}
    for arm, name in authority_names.items():
        authority_path = docs / name
        authority = baseline.load_json(authority_path)
        if authority.get("status") != "ESTABLISHED":
            raise baseline.EvaluationAuthorityError(
                f"{arm} metric authority is not established"
            )
        if authority.get("include_hidden") is not True:
            raise baseline.EvaluationAuthorityError(
                f"{arm} hidden policy mismatch"
            )
        checks: dict[str, str] = {}
        for field, filename in field_files.items():
            actual = baseline.sha256_file(result_root / filename)
            if actual != authority[field]:
                raise baseline.EvaluationAuthorityError(
                    f"{arm} baseline metric hash mismatch: {filename}"
                )
            checks[filename] = actual
        records[arm] = {
            "authority_path": str(authority_path),
            "authority_sha256": baseline.sha256_file(authority_path),
            "table_hashes": checks,
        }
    aggregate_path = (
        result_root / "B0_B1_R0_STANDARD_V2_AGGREGATE_METRICS.csv"
    )
    expected_aggregate = (
        "78fabac0df362f241594c5c55ec73d6a9028bb87c3e92026a96393b349db3c33"
    )
    if baseline.sha256_file(aggregate_path) != expected_aggregate:
        raise baseline.EvaluationAuthorityError(
            "Baseline aggregate metric authority mismatch"
        )
    return {
        "status": "PASS",
        "result_root": str(result_root),
        "aggregate_sha256": expected_aggregate,
        "arms": records,
    }


def preflight(
    source_repo: Path,
    worktree_repo: Path,
) -> tuple[list[dict[str, Any]], baseline.ArmSpec, dict[str, Any]]:
    """Validate all frozen predictions and the common four-arm population."""

    videos, _, _, base_state = baseline.preflight(
        source_repo,
        worktree_repo,
    )
    arm = r1_spec(source_repo, worktree_repo)
    authority = baseline.load_json(arm.authority_path)
    if authority.get("status") != "ESTABLISHED":
        raise baseline.EvaluationAuthorityError("R1 authority not established")
    if authority.get("r1_prediction_artifact_sha256") != R1_ARTIFACT_SHA256:
        raise baseline.EvaluationAuthorityError("R1 authority hash mismatch")
    if authority.get("repair_semantic_sha256") != R1_REPAIR_SEMANTIC_SHA256:
        raise baseline.EvaluationAuthorityError(
            "R1 repair semantic hash mismatch"
        )
    marker = arm.prediction_root.parent / (
        "FROZEN_SCIENTIFIC_AUTHORITY_DO_NOT_DELETE.txt"
    )
    if not marker.is_file():
        raise baseline.EvaluationAuthorityError("R1 retention marker missing")
    manifest_path = (
        worktree_repo
        / "docs"
        / "tracking"
        / "r1_prediction_authority"
        / "R1_RF_HYBRID_OFFLINE_PREDICTION_ARTIFACT_MANIFEST_20260728.json"
    )
    manifest = baseline.load_json(manifest_path)
    if manifest.get("canonical_prediction_content_sha256") != (
        R1_ARTIFACT_SHA256
    ):
        raise baseline.EvaluationAuthorityError("R1 manifest hash mismatch")
    expected = {row["video_key"]: row for row in manifest["predictions"]}
    if len(expected) != baseline.EXPECTED_VIDEOS:
        raise baseline.EvaluationAuthorityError("R1 video count mismatch")
    records: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    before = dict(base_state["prediction_hashes"])
    for video in videos:
        video_key = video["video_key"]
        path = arm.prediction_root / f"{video_key}.xml"
        expected_row = expected.get(video_key)
        if expected_row is None or not path.is_file():
            raise baseline.EvaluationAuthorityError(
                f"Missing R1 prediction: {video_key}"
            )
        actual_sha = baseline.sha256_file(path)
        if actual_sha != expected_row["sha256"]:
            raise baseline.EvaluationAuthorityError(
                f"R1 prediction hash mismatch: {video_key}"
            )
        metadata = baseline.video_metadata(Path(video["source_video_path"]))
        record = baseline.prediction_structural_record(
            path,
            video_key=video_key,
            width=int(metadata["video_width"]),
            height=int(metadata["video_height"]),
        )
        if (
            record["prediction_semantic_sha256"]
            != expected_row["semantic_sha256"]
            or record["canonical_row_sha256"]
            != expected_row["canonical_row_sha256"]
        ):
            raise baseline.EvaluationAuthorityError(
                f"R1 semantic prediction mismatch: {video_key}"
            )
        records.append(record)
        before[str(path)] = actual_sha
        audit = baseline.adapter_audit(path)
        adapter_rows.append({"arm": "R1", "video_key": video_key, **audit})
        video["prediction_paths"]["R1"] = str(path)
        video["prediction_coverage"] = {
            "B0": "PASS",
            "B1": "PASS",
            "R0": "PASS",
            "R1": "PASS",
        }
    if baseline.prediction_set_hash(records) != R1_ARTIFACT_SHA256:
        raise baseline.EvaluationAuthorityError(
            "R1 prediction-set authority mismatch"
        )
    return videos, arm, {
        "prediction_hashes": before,
        "R1_adapter_rows": adapter_rows,
        "R1_manifest_sha256": baseline.sha256_file(manifest_path),
        "R1_prediction_records": records,
    }


def evaluate_r1_pass(
    pass_root: Path,
    arm: baseline.ArmSpec,
    videos: list[dict[str, Any]],
    *,
    evaluator_code_sha: str,
    reverse_inputs: bool,
) -> dict[str, Any]:
    """Evaluate R1 through the established implementation and rename outputs."""

    result = baseline.evaluate_pass(
        pass_root,
        (arm,),
        videos,
        evaluator_code_sha=evaluator_code_sha,
        reverse_inputs=reverse_inputs,
    )
    for old_name, new_name in R1_FILE_MAP.items():
        (pass_root / old_name).replace(pass_root / new_name)
    result["output_hashes"] = {
        new_name: baseline.sha256_file(pass_root / new_name)
        for new_name in R1_FILE_MAP.values()
    }
    conservation_path = pass_root / "IDENTITY_EVENT_CONSERVATION.json"
    conservation_path.replace(pass_root / "R1_IDENTITY_EVENT_CONSERVATION.json")
    return result


def compare_r1_passes(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
) -> dict[str, Any]:
    """Require byte-identical R1 authority outputs."""

    checks = {
        name: {
            "pass1_sha256": pass1["output_hashes"][name],
            "pass2_sha256": pass2["output_hashes"][name],
            "equal": pass1["output_hashes"][name]
            == pass2["output_hashes"][name],
        }
        for name in REPEAT_FILES
    }
    passed = all(item["equal"] for item in checks.values())
    return {
        "schema_version": "tracking.development_2x2.r1_determinism.v1",
        "date": DATE,
        "complete_evaluation_passes": 2,
        "pass2_prediction_file_order": "REVERSED",
        "pass2_prediction_row_order": "REVERSED_WITHIN_FRAME",
        "required_output_files": checks,
        "R1_reevaluation_repeatability": "PASS" if passed else "FAIL",
        "R1_input_order_invariance": "PASS" if passed else "FAIL",
    }


def _effect_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    values = aggregate.set_index("arm")
    rows = []
    for metric, role, higher, unit in METRICS:
        arm_values = {
            arm: float(values.loc[arm, metric])
            for arm in ("B0", "B1", "R0", "R1")
        }
        b_effect = arm_values["B1"] - arm_values["B0"]
        r_effect = arm_values["R1"] - arm_values["R0"]
        rows.append(
            {
                "metric": metric,
                "metric_role": role,
                "higher_is_better": higher,
                **arm_values,
                "B1_minus_B0": b_effect,
                "R1_minus_R0": r_effect,
                "R0_minus_B0": arm_values["R0"] - arm_values["B0"],
                "R1_minus_B1": arm_values["R1"] - arm_values["B1"],
                "interaction_raw": r_effect - b_effect,
                "bytetrack_repair_oriented_effect": (
                    b_effect if higher else -b_effect
                ),
                "rf_repair_oriented_effect": (
                    r_effect if higher else -r_effect
                ),
                "interaction_oriented_effect": (
                    r_effect - b_effect if higher else b_effect - r_effect
                ),
                "unit": unit,
                "interpretation_scope": (
                    "REPAIR_BY_COMPLETE_CORE_PIPELINE_INCLUDING_"
                    "PROFILE_SPECIFIC_DETECTOR_CADENCE"
                ),
            }
        )
    return pd.DataFrame(rows)


def _wide_effects(
    table: pd.DataFrame,
    keys: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key_values, group in table.groupby(keys, sort=True, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        indexed = group.set_index("arm")
        for metric in metrics:
            values = {
                arm: float(indexed.loc[arm, metric])
                for arm in ("B0", "B1", "R0", "R1")
            }
            rows.append(
                {
                    **dict(zip(keys, key_values, strict=True)),
                    "metric": metric,
                    **values,
                    "B1_minus_B0": values["B1"] - values["B0"],
                    "R1_minus_R0": values["R1"] - values["R0"],
                    "R0_minus_B0": values["R0"] - values["B0"],
                    "R1_minus_B1": values["R1"] - values["B1"],
                    "interaction": (
                        values["R1"]
                        - values["R0"]
                        - values["B1"]
                        + values["B0"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_summary(per_video_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in ("B1_minus_B0", "R1_minus_R0"):
        for metric, role, higher, unit in METRICS:
            values = per_video_effects.loc[
                per_video_effects["metric"] == metric, contrast
            ]
            oriented = values if higher else -values
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "metric_role": role,
                    "unit": unit,
                    "video_count": len(values),
                    "median_paired_difference": float(values.median()),
                    "q1_paired_difference": float(values.quantile(0.25)),
                    "q3_paired_difference": float(values.quantile(0.75)),
                    "iqr_paired_difference": float(
                        values.quantile(0.75) - values.quantile(0.25)
                    ),
                    "improving_videos": int((oriented > 0).sum()),
                    "worsening_videos": int((oriented < 0).sum()),
                    "tied_videos": int((oriented == 0).sum()),
                    "worst_oriented_change": float(oriented.min()),
                    "best_oriented_change": float(oriented.max()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_idf1(per_video: pd.DataFrame) -> pd.DataFrame:
    """Cluster-bootstrap IDF1 from valid additive identity counts."""

    rng = np.random.default_rng(20260728)
    video_keys = sorted(set(per_video["video_stem"]))
    indexed = {
        arm: per_video.loc[per_video["arm"] == arm].set_index("video_stem")
        for arm in ("B0", "B1", "R0", "R1")
    }
    rows = []
    for left, right, name in (
        ("B0", "B1", "B1_minus_B0"),
        ("R0", "R1", "R1_minus_R0"),
    ):
        effects = np.empty(10000, dtype=float)
        for index in range(10000):
            sample = rng.choice(video_keys, size=len(video_keys), replace=True)
            values = {}
            for arm in (left, right):
                selected = indexed[arm].loc[list(sample)]
                idtp = float(selected["idtp"].sum())
                idfp = float(selected["idfp"].sum())
                idfn = float(selected["idfn"].sum())
                values[arm] = 2 * idtp / (2 * idtp + idfp + idfn)
            effects[index] = values[right] - values[left]
        rows.append(
            {
                "contrast": name,
                "metric": "idf1",
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 20260728,
                "lower_95": float(np.quantile(effects, 0.025)),
                "upper_95": float(np.quantile(effects, 0.975)),
                "status": "COMPUTED_FROM_ADDITIVE_IDENTITY_COUNTS",
            }
        )
        rows.append(
            {
                "contrast": name,
                "metric": "hota",
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 20260728,
                "lower_95": None,
                "upper_95": None,
                "status": "NOT_COMPUTED_CONTRACT_UNAVAILABLE",
            }
        )
    return pd.DataFrame(rows)


def classify_repair(
    aggregate: pd.DataFrame,
    repaired: str,
    raw: str,
) -> str:
    """Apply the predeclared conservative repair-effect classification."""

    rows = aggregate.set_index("arm")
    hota = float(rows.loc[repaired, "hota"] - rows.loc[raw, "hota"])
    idf1 = float(rows.loc[repaired, "idf1"] - rows.loc[raw, "idf1"])
    severity = {
        metric: float(rows.loc[repaired, metric] - rows.loc[raw, metric])
        for metric in (
            "wrong_id_matched_frames",
            "terminal_identity_error_episode_count",
            "persistent_pairwise_identity_swap_count",
        )
    }
    if hota > 0 and idf1 > 0 and all(value <= 0 for value in severity.values()):
        return "BROADLY_BENEFICIAL"
    if hota < 0 and idf1 < 0 and any(value > 0 for value in severity.values()):
        return "BROADLY_HARMFUL"
    if (
        abs(hota) <= 1e-15
        and abs(idf1) <= 1e-15
        and all(value == 0 for value in severity.values())
    ):
        return "NO_OBSERVED_CHANGE"
    return "MIXED_TRADEOFF"


def _raw_shapes(path: Path) -> dict[int, list[TrackingObject]]:
    payload = baseline.load_json(path)
    frames: dict[int, list[TrackingObject]] = defaultdict(list)
    for shape in payload["shapes"]:
        if bool(shape.get("outside", False)):
            continue
        attrs = {
            str(item.get("name")): str(item.get("value", ""))
            for item in shape.get("attributes", [])
        }
        frame = int(shape["frame"])
        frames[frame].append(
            TrackingObject(
                frame=frame,
                obj_id=attrs["ID"],
                bbox=tuple(float(value) for value in shape["points"]),
                hidden=attrs.get("Hidden", "No").lower() == "yes",
                source_track_id=str(shape.get("_raw_track_id") or ""),
                label=str(shape.get("label", "")),
            )
        )
    return dict(frames)


def _frame_track_scores(
    gt: dict[int, list[TrackingObject]],
    pred: dict[int, list[TrackingObject]],
    authority: dict[str, str],
) -> dict[tuple[int, str], int]:
    scores: dict[tuple[int, str], int] = {}
    for frame in sorted(set(gt) | set(pred)):
        gt_objects = gt.get(frame, [])
        pred_objects = pred.get(frame, [])
        matches = match_frame_eligible(
            gt_objects,
            pred_objects,
            iou_threshold=0.5,
        )
        matched_pred: set[int] = set()
        for gt_index, pred_index, _ in matches:
            matched_pred.add(pred_index)
            gt_id = gt_objects[gt_index].obj_id
            pred_id = pred_objects[pred_index].obj_id
            scores[(frame, pred_id)] = 2 if authority.get(gt_id) == pred_id else 1
        for pred_index, item in enumerate(pred_objects):
            if pred_index not in matched_pred:
                scores[(frame, item.obj_id)] = 0
    return scores


def r1_event_attribution(
    source_repo: Path,
    videos: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Conservatively attribute R1 ledger events without double claims."""

    root = (
        source_repo
        / "outputs"
        / "tracking"
        / "frozen_predictions_standard_v2_20260728_retry1"
        / "R1_rf_hybrid_offline"
    )
    authority_table = pd.read_csv(
        source_repo
        / BASELINE_RESULT_RELATIVE
        / "B0_B1_R0_IDENTITY_AUTHORITIES.csv"
    )
    authority_table = authority_table.loc[authority_table["arm"] == "R0"]
    rows = []
    claimed: set[tuple[str, int, str]] = set()
    double_count = 0
    frames_improved = 0
    frames_harmed = 0
    for video in videos:
        key = video["video_key"]
        gt = parse_cvat_video_xml(
            Path(video["gt_path"]),
            include_hidden=True,
            start_frame=0,
            end_frame=baseline.EXPECTED_FRAMES - 1,
        )
        raw = _raw_shapes(
            root / "raw_core_snapshots" / f"{key}.rf_raw_track_output.json"
        )
        repaired = parse_cvat_video_xml(
            root / "predictions" / f"{key}.xml",
            include_hidden=True,
            start_frame=0,
            end_frame=baseline.EXPECTED_FRAMES - 1,
        )
        video_authority = authority_table.loc[
            authority_table["sequence_key"] == key
        ]
        mapping = dict(
            zip(
                video_authority["gt_id"].astype(str),
                video_authority["pred_id"].astype(str),
                strict=True,
            )
        )
        raw_scores = _frame_track_scores(gt, raw, mapping)
        repaired_scores = _frame_track_scores(gt, repaired, mapping)
        ledger = baseline.load_json(
            root
            / "repair_ledgers"
            / f"{key}.rf_offline_repair_ledger.json"
        )
        for event in sorted(
            ledger["events"],
            key=lambda item: item["repair_event_id"],
        ):
            track_id = str(event["output_track_id"])
            event_keys = {
                (key, frame, track_id)
                for frame in range(
                    int(event["start_frame"]),
                    int(event["end_frame"]) + 1,
                )
            }
            overlap = event_keys & claimed
            if overlap:
                double_count += len(overlap)
                outcome = "AMBIGUOUS_OVERLAP"
                improved = 0
                harmed = 0
            else:
                claimed.update(event_keys)
                improved = sum(
                    repaired_scores.get((frame, track_id), 0)
                    > raw_scores.get(
                        (frame, str(event["input_track_id"])),
                        0,
                    )
                    for _, frame, _ in event_keys
                )
                harmed = sum(
                    repaired_scores.get((frame, track_id), 0)
                    < raw_scores.get(
                        (frame, str(event["input_track_id"])),
                        0,
                    )
                    for _, frame, _ in event_keys
                )
                if improved and not harmed:
                    outcome = "BENEFICIAL"
                elif harmed and not improved:
                    outcome = "HARMFUL"
                elif improved or harmed:
                    outcome = "AMBIGUOUS_OVERLAP"
                else:
                    outcome = "NEUTRAL"
            frames_improved += improved
            frames_harmed += harmed
            rows.append(
                {
                    **event,
                    "attribution_outcome": outcome,
                    "authoritative_frames_improved": improved,
                    "authoritative_frames_harmed": harmed,
                    "attribution_role": "AVAILABLE_DIAGNOSTIC_ONLY",
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["video_key", "repair_event_id"],
        kind="stable",
    )
    counts = Counter(table["attribution_outcome"])
    conservation = {
        "schema_version": "tracking.development_2x2.repair_attribution.v1",
        "date": DATE,
        "repair_attribution_status": (
            "PARTIAL_R1_ONLY_B1_FROZEN_LEDGER_UNAVAILABLE"
        ),
        "B1_repair_event_attribution": (
            "NOT_EVALUABLE_FROZEN_LEDGER_UNAVAILABLE"
        ),
        "B1_repair_events_total": "NOT_AVAILABLE",
        "R1_repair_event_attribution": "AVAILABLE_DIAGNOSTIC_ONLY",
        "R1_repair_events_total": len(table),
        "R1_outcome_counts": dict(sorted(counts.items())),
        "R1_frames_improved": frames_improved,
        "R1_frames_harmed": frames_harmed,
        "repair_attribution_double_count": double_count,
        "cross_core_event_attribution_comparison_authorized": False,
        "repair_event_attribution_required_for_2x2_metric_authority": False,
        "B1_attribution_limitation_blocks_2x2_authority": False,
    }
    return table, conservation


def _population_document(videos: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for video in videos:
        rows.append(
            {
                key: video[key]
                for key in (
                    "video_key",
                    "source_video_sha256",
                    "gt_sha256",
                    "frame_start",
                    "frame_end",
                    "frame_count",
                    "visible_gt_rows",
                    "hidden_gt_rows",
                    "sequence_boundary",
                    "aggregate_inclusion_status",
                    "gt_authority_status",
                    "mechanism_ranking_eligibility",
                    "prediction_coverage",
                )
            }
            | {
                "detector_cadence": {
                    "B0": "EVERY_FRAME",
                    "B1": "EVERY_FRAME",
                    "R0": "EVERY_2_FRAMES",
                    "R1": "EVERY_2_FRAMES",
                }
            }
        )
    return {
        "schema_version": "tracking.development_2x2.population.v1",
        "date": DATE,
        "video_count": len(rows),
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_gt_authority": "PASS",
        "common_source_video_authority": "PASS",
        "common_sequence_boundary_authority": "PASS",
        "videos": rows,
    }


def freeze(
    source_repo: Path,
    worktree_repo: Path,
    result_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    """Execute the R1-only evaluation and freeze the complete 2x2 authority."""

    if result_root.exists():
        raise baseline.EvaluationAuthorityError(
            f"Refusing overwrite: {result_root}"
        )
    baseline_authority = validate_baseline_metric_authority(
        source_repo,
        worktree_repo,
    )
    videos, arm, state = preflight(source_repo, worktree_repo)
    result_root.mkdir(parents=True)
    code_sha = resolve_evaluator_code_sha()
    if code_sha != baseline.git_sha(worktree_repo):
        raise baseline.EvaluationAuthorityError(
            "Evaluator code SHA is not current clean orchestration HEAD"
        )
    pass1 = evaluate_r1_pass(
        result_root / "pass1",
        arm,
        videos,
        evaluator_code_sha=code_sha,
        reverse_inputs=False,
    )
    pass2 = evaluate_r1_pass(
        result_root / "pass2",
        arm,
        videos,
        evaluator_code_sha=code_sha,
        reverse_inputs=True,
    )
    determinism = compare_r1_passes(pass1, pass2)
    if determinism["R1_reevaluation_repeatability"] != "PASS":
        raise baseline.EvaluationAuthorityError("R1 determinism failed")
    conservation = pass1["conservation"]
    if (
        conservation["wrong_id_row_conservation"] != "PASS"
        or conservation["tp_fp_fn_conservation"] != "PASS"
        or conservation["identity_episode_double_count"] != 0
        or conservation["pairwise_swap_double_count"] != 0
    ):
        raise baseline.EvaluationAuthorityError("R1 conservation failed")
    for path in sorted((result_root / "pass1").iterdir()):
        if path.is_file():
            shutil.copy2(path, result_root / path.name)
    baseline_root = source_repo / BASELINE_RESULT_RELATIVE
    aggregate = pd.concat(
        [
            pd.read_csv(
                baseline_root / "B0_B1_R0_STANDARD_V2_AGGREGATE_METRICS.csv"
            ),
            pd.read_csv(result_root / REPEAT_FILES[0]),
        ],
        ignore_index=True,
    ).sort_values("arm", kind="stable")
    per_video = pd.concat(
        [
            pd.read_csv(
                baseline_root / "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv"
            ),
            pd.read_csv(result_root / REPEAT_FILES[1]),
        ],
        ignore_index=True,
    ).sort_values(["arm", "video_stem"], kind="stable")
    per_alpha = pd.concat(
        [
            pd.read_csv(
                baseline_root / "B0_B1_R0_STANDARD_V2_PER_ALPHA_METRICS.csv"
            ),
            pd.read_csv(result_root / REPEAT_FILES[2]),
        ],
        ignore_index=True,
    ).sort_values(["arm", "video_stem", "alpha"], kind="stable")
    effects = _effect_table(aggregate)
    per_video_effects = _wide_effects(
        per_video,
        ["video_stem"],
        tuple(item[0] for item in METRICS),
    )
    per_alpha_effects = _wide_effects(
        per_alpha,
        ["video_stem", "alpha"],
        ("hota", "deta", "assa", "loca"),
    )
    paired = _paired_summary(per_video_effects)
    bootstrap = _bootstrap_idf1(per_video)
    bytetrack_class = classify_repair(aggregate, "B1", "B0")
    rf_class = classify_repair(aggregate, "R1", "R0")
    attribution, attribution_conservation = r1_event_attribution(
        source_repo,
        videos,
    )
    baseline.write_csv(
        effects,
        result_root / f"DEVELOPMENT_2X2_AGGREGATE_EFFECTS_{DATE}.csv",
    )
    baseline.write_csv(
        per_video_effects,
        result_root / f"DEVELOPMENT_2X2_PER_VIDEO_EFFECTS_{DATE}.csv",
    )
    baseline.write_csv(
        per_alpha_effects,
        result_root / f"DEVELOPMENT_2X2_PER_ALPHA_EFFECTS_{DATE}.csv",
    )
    baseline.write_csv(
        paired,
        result_root / f"DEVELOPMENT_2X2_PAIRED_VIDEO_SUMMARY_{DATE}.csv",
    )
    baseline.write_csv(
        bootstrap,
        result_root / f"DEVELOPMENT_2X2_BOOTSTRAP_INTERVALS_{DATE}.csv",
    )
    baseline.write_csv(
        attribution,
        result_root / f"DEVELOPMENT_2X2_REPAIR_EVENT_OUTCOMES_{DATE}.csv",
    )
    baseline.write_json(
        result_root / f"DEVELOPMENT_2X2_REPAIR_EVENT_CONSERVATION_{DATE}.json",
        attribution_conservation,
    )
    classification = {
        "schema_version": "tracking.development_2x2.classification.v1",
        "date": DATE,
        "bytetrack_repair_effect_classification": bytetrack_class,
        "rf_repair_effect_classification": rf_class,
        "classification_source": "PREDECLARED_METRIC_HIERARCHY",
        "R1_event_attribution_used_for_repair_effect_classification": False,
    }
    baseline.write_json(
        result_root
        / f"DEVELOPMENT_2X2_REPAIR_EFFECT_CLASSIFICATION_{DATE}.json",
        classification,
    )
    post = baseline._post_prediction_hashes(state["prediction_hashes"])
    if post["prediction_artifacts_modified"] != 0:
        raise baseline.EvaluationAuthorityError("Prediction mutation detected")
    mp4_count = sum(
        path.suffix.lower() == ".mp4"
        for path in result_root.rglob("*")
        if path.is_file()
    )
    if mp4_count:
        raise baseline.EvaluationAuthorityError("MP4 found in result root")
    aggregate_hash = baseline.sha256_file(
        result_root / f"DEVELOPMENT_2X2_AGGREGATE_EFFECTS_{DATE}.csv"
    )
    per_video_hash = baseline.sha256_file(
        result_root / f"DEVELOPMENT_2X2_PER_VIDEO_EFFECTS_{DATE}.csv"
    )
    per_alpha_hash = baseline.sha256_file(
        result_root / f"DEVELOPMENT_2X2_PER_ALPHA_EFFECTS_{DATE}.csv"
    )
    attribution_hash = baseline.sha256_file(
        result_root / f"DEVELOPMENT_2X2_REPAIR_EVENT_OUTCOMES_{DATE}.csv"
    )
    r1_hashes = {
        name: baseline.sha256_file(result_root / name)
        for name in REPEAT_FILES
    }
    population = _population_document(videos)
    docs_root.mkdir(parents=True, exist_ok=True)
    baseline.write_json(
        docs_root / f"DEVELOPMENT_2X2_POPULATION_MANIFEST_{DATE}.json",
        population,
    )
    baseline.write_json(
        docs_root
        / f"DEVELOPMENT_2X2_REPAIR_EFFECT_CLASSIFICATION_{DATE}.json",
        classification,
    )
    baseline.write_json(
        docs_root
        / f"DEVELOPMENT_2X2_REPAIR_EVENT_CONSERVATION_{DATE}.json",
        attribution_conservation,
    )
    r1_row = aggregate.loc[aggregate["arm"] == "R1"].iloc[0].to_dict()
    authority = {
        "schema_version": "tracking.development_2x2.authority.v1",
        "date": DATE,
        "status": "ESTABLISHED",
        "prediction_artifact_sha256": {
            "B0": (
                "13d9226c36141264cc33e4b498d38e5f3eaa9891cf32bc4c8"
                "fb87b01fd27d576"
            ),
            "B1": (
                "569c49e00905add068fac70c919fe21c10127e3ab773528a4ac44"
                "199fcb4835b"
            ),
            "R0": (
                "fd2d4f3dec0710d1c9eecba9308247a7b226dd34a4a02a9cb89"
                "f17acb22bbbfe"
            ),
            "R1": R1_ARTIFACT_SHA256,
        },
        "baseline_metric_authority": baseline_authority,
        "R1_standard_v2_authority": "ESTABLISHED",
        "R1_aggregate_metrics": r1_row,
        "R1_evaluation_hashes": r1_hashes,
        "aggregate_effect_table_sha256": aggregate_hash,
        "per_video_effect_table_sha256": per_video_hash,
        "per_alpha_effect_table_sha256": per_alpha_hash,
        "repair_attribution_table_sha256": attribution_hash,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "primary_include_hidden": True,
        "evaluator_code_sha": code_sha,
        "metric_config_sha256": str(r1_row["metric_config_sha256"]),
        "R1_determinism": determinism,
        "R1_conservation": conservation,
        "repair_attribution_status": (
            "PARTIAL_R1_ONLY_B1_FROZEN_LEDGER_UNAVAILABLE"
        ),
        "scientific_limitations": [
            "Development population only.",
            "B1 has no frozen raw pre-repair output or repair ledger.",
            "Cross-core effects include profile-specific detector cadence.",
            "R1 is post-video offline and cannot support a realtime claim.",
            "No unseen evaluation or promotion is authorized.",
        ],
    }
    decision = {
        "schema_version": "tracking.development_2x2.decision.v1",
        "date": DATE,
        "decision": "PASS_COMPLETE_DEVELOPMENT_2X2_AUTHORITY_ESTABLISHED",
        "development_2x2_authority": "ESTABLISHED",
        "B0_metric_authority_reused": True,
        "B1_metric_authority_reused": True,
        "R0_metric_authority_reused": True,
        "R1_standard_v2_authority": "ESTABLISHED",
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_gt_authority": "PASS",
        "common_evaluator_authority": "PASS",
        "common_hidden_policy": "PASS",
        "wrong_id_row_conservation": "PASS",
        "tp_fp_fn_conservation": "PASS",
        "R1_reevaluation_repeatability": "PASS",
        "R1_input_order_invariance": "PASS",
        "prediction_artifacts_modified": 0,
        "tracker_executions": 0,
        "detector_inference_calls": 0,
        "unseen_videos_accessed": False,
        "repair_attribution_status": (
            "PARTIAL_R1_ONLY_B1_FROZEN_LEDGER_UNAVAILABLE"
        ),
        "ready_for_unseen_method_freeze_decision": True,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
    }
    baseline.write_json(
        docs_root / f"DEVELOPMENT_2X2_STANDARD_V2_AUTHORITY_{DATE}.json",
        authority,
    )
    baseline.write_json(
        docs_root / f"DEVELOPMENT_2X2_STANDARD_V2_DECISION_{DATE}.json",
        decision,
    )
    run_manifest = {
        "schema_version": "tracking.development_2x2.run_manifest.v1",
        "date": DATE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "evaluation_code_sha": code_sha,
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "scientific-ablation-controller",
        ],
        "result_root": str(result_root),
        "R1_complete_evaluation_passes": 2,
        "B0_B1_R0_metric_authority_reused": True,
        "prediction_artifacts_before_and_after": post,
        "execution_counts": {
            "tracker_executions": 0,
            "detector_inference_calls": 0,
            "prediction_artifacts_regenerated": 0,
            "prediction_artifacts_modified": 0,
            "run_root_mp4_count": 0,
            "unseen_videos_accessed": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    baseline.write_json(
        result_root / "DEVELOPMENT_2X2_STANDARD_V2_RUN_MANIFEST.json",
        run_manifest,
    )
    baseline.write_json(
        result_root / "R1_STANDARD_V2_DETERMINISM.json",
        determinism,
    )
    inventory = baseline.artifact_inventory(result_root)
    baseline.write_json(
        result_root / "DEVELOPMENT_2X2_STANDARD_V2_ARTIFACT_INVENTORY.json",
        {
            "schema_version": "tracking.development_2x2.inventory.v1",
            "date": DATE,
            "artifact_count": len(inventory),
            "artifacts": inventory,
            "canonical_inventory_sha256": baseline.canonical_hash(inventory),
        },
    )
    marker = result_root / "FROZEN_SCIENTIFIC_2X2_AUTHORITY_DO_NOT_DELETE.txt"
    marker.write_text(
        "NON_DISPOSABLE_FROZEN_DEVELOPMENT_2X2_STANDARD_V2_AUTHORITY\n",
        encoding="utf-8",
        newline="\n",
    )
    for path in result_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    return {
        "decision": decision,
        "R1_aggregate_metrics": r1_row,
        "classification": classification,
        "attribution": attribution_conservation,
        "result_root": str(result_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "tracking"
            / "development_2x2_standard_v2"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_repo = args.source_repo.resolve()
    worktree_repo = args.worktree_repo.resolve()
    if args.preflight_only:
        baseline_authority = validate_baseline_metric_authority(
            source_repo,
            worktree_repo,
        )
        videos, _, state = preflight(source_repo, worktree_repo)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "video_count": len(videos),
                    "baseline_metric_authority": baseline_authority["status"],
                    "R1_prediction_file_count": len(
                        state["R1_prediction_records"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = freeze(
        source_repo,
        worktree_repo,
        args.result_root.resolve(),
        args.docs_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
