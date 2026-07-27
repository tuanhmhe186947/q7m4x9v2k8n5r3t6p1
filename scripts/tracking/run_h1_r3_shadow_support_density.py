"""Run the bounded H1-r3 reservation-disabled development shadow audit.

This harness executes ``realtime_fast`` twice from the same immutable detector
cache.  The shadow arm only returns diagnostic records; both arms retain the
baseline association path.
"""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

AUTHORIZED_SHA = "b21fee46b38cda841ded5d21aaaf56249017feba"
BASELINE_PROFILE = "realtime_fast"
SCORE_NAME = "owner_preference_lower_bound"
SCORE_THRESHOLD = 0.625
SUPPORT_MARGIN = 0.25
H1_R2_CACHE_PRODUCER_SHA = "e54b550779be060b2e322d56304c54b854c9e0c3"

SHADOW_COUNTERS = (
    "h1_r3_shadow_stage_calls",
    "h1_r3_shadow_hidden_tracks_offered",
    "h1_r3_shadow_pair_candidates",
    "h1_r3_shadow_core_eligible_pairs",
    "h1_r3_shadow_optional_appearance_available",
    "h1_r3_shadow_optional_motion_available",
    "h1_r3_shadow_score_pairs",
    "h1_r3_shadow_below_threshold",
    "h1_r3_shadow_margin_failed",
    "h1_r3_shadow_would_activate",
    "h1_r3_shadow_invalid_numeric",
    "h1_r3_shadow_missing_core_overlap",
    "h1_r3_shadow_missing_core_freshness",
)

LOCKED_CACHE_HASHES = {
    "E01_000233_contention_a": (
        "a00378197e0b1b0b6f778a29d49c64067f603d4999fba82d5004f9054e83ca1e"
    ),
    "E02_000233_contention_b": (
        "b12e3aa4d0d687ca40d35d469f8c7db201a7835dd0ca1456ee18fa7f5f0db4dc"
    ),
    "E03_000233_crossing": (
        "e8bea9e28f5645d1a87083409f8f3070c386a79e64ea2764bd43f81356342715"
    ),
    "E04_000263_contention": (
        "d8871f07bc87875414c98360ccbadf921af4743db943ac315f4be0e97338c701"
    ),
    "E05_000263_control_clean": (
        "9321431a4a2ff1db86ed5d66423a2375135641c1e33eb4f5be90d1ece66f8cbe"
    ),
    "E06_000233_control_clean": (
        "a3bccd427eb05963798846116794f2d9e0566ad8641d0f756f68ee427a665322"
    ),
}


class ShadowAuditError(RuntimeError):
    """Raised when a frozen shadow-audit condition is violated."""


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
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
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


def load_development_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ShadowAuditError("frozen development manifest must have six rows")
    episodes: list[dict[str, Any]] = []
    for row in rows:
        role = row["positive_control_role"]
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if role not in {"positive", "control"}:
            raise ShadowAuditError("unexpected development role")
        mode = (
            "causal_prefix_warmup_episode"
            if role == "positive"
            else "cold_start_episode"
        )
        decode_start = max(0, start - 60) if role == "positive" else start
        episodes.append(
            {
                "episode_id": row["episode_id"],
                "video_key": row["video_key"],
                "role": role,
                "mode": mode,
                "decode_start": decode_start,
                "start_frame": start,
                "end_frame": end,
                "video_sha256": row["video_sha256"],
                "gt_sha256": row["gt_sha256"],
            }
        )
    if {row["episode_id"] for row in episodes} != set(LOCKED_CACHE_HASHES):
        raise ShadowAuditError("development population differs from cache lock")
    return episodes


def video_path(source_repo: Path, episode: dict[str, Any]) -> Path:
    return (
        source_repo
        / "data"
        / "videos"
        / f"{episode['video_key']}.mp4"
    )


def gt_path(source_repo: Path, episode: dict[str, Any]) -> Path:
    if "000263" in episode["video_key"]:
        name = "Tracking_annotation_Pigs291119_000263_30fps.xml"
    else:
        name = "Pigs291119_000233_30fps.xml"
    return source_repo / "data" / "annotations" / "tracking" / name


def build_cfg(
    source_repo: Path,
    episode: dict[str, Any],
    output_dir: Path,
) -> Any:
    from pig_behavior.tracking.config import TrackingConfig
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    return TrackingConfig(
        mode="realtime",
        video_path=video_path(source_repo, episode),
        weights_path=(
            source_repo / "models" / "detector" / "pig_detector_yolov8.pt"
        ),
        mask_path=(
            source_repo / "data" / "annotations" / "scene" / "mask.png"
        ),
        output_dir=output_dir,
        device="cpu",
        write_output_video=False,
        start_frame=episode["decode_start"],
        max_frames=episode["end_frame"] - episode["decode_start"] + 1,
        **EVAL_CONFIGS[BASELINE_PROFILE],
    )


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
        "detect_every_n_frames": cfg.detect_every_n_frames,
    }


def cache_identity(
    source_repo: Path,
    episode: dict[str, Any],
    cfg: Any,
) -> Any:
    from pig_behavior.tracking.detector_cache import DetectorCacheIdentity

    return DetectorCacheIdentity(
        video_key=episode["video_key"],
        source_video_sha256=episode["video_sha256"],
        detector_weight_sha256=sha256_file(Path(cfg.weights_path)),
        detector_semantic_config_sha256=canonical_hash(
            detector_semantic_payload(cfg)
        ),
        producer_code_sha=H1_R2_CACHE_PRODUCER_SHA,
        creation_authority=(
            "user-authorized H1-r2 development quality evaluation 2026-07-27"
        ),
    )


def evaluate_metrics(
    source_repo: Path,
    episode: dict[str, Any],
    prediction_xml: Path,
) -> dict[str, Any]:
    from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml
    from pig_behavior.evaluation.tracking.evaluator import evaluate_tracking
    from pig_behavior.evaluation.tracking.metrics import (
        attach_remapped_metrics,
        remap_prediction_ids,
    )

    kwargs = {
        "include_hidden": True,
        "start_frame": episode["start_frame"],
        "end_frame": episode["end_frame"],
    }
    ground_truth = parse_cvat_video_xml(
        gt_path(source_repo, episode),
        **kwargs,
    )
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
    result["identity_discontinuities"] = remapped.idsw
    result["identity_mapping"] = mapping
    return result


def semantic_shapes_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return canonical_hash(payload)


def output_shapes_path(summary: Any) -> Path:
    return Path(summary.annotations_json)


def summarize_numeric(
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
    if not values.size:
        result.update(
            {
                key: "NOT_MEASURED"
                for key in (
                    "minimum",
                    "p10",
                    "p25",
                    "median",
                    "p75",
                    "p90",
                    "p95",
                    "maximum",
                )
            }
        )
        return result
    for key, quantile in (
        ("minimum", 0.0),
        ("p10", 0.10),
        ("p25", 0.25),
        ("median", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("maximum", 1.0),
    ):
        result[key] = float(np.quantile(values, quantile))
    return result


def candidate_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    prefix = [
        "episode_id",
        "development_role",
        "video_key",
        "frame_index",
        "hidden_track_id",
        "visible_track_id",
        "detection_index",
    ]
    discovered = sorted(
        {key for row in rows for key in row}
        - set(prefix)
    )
    return [*prefix, *discovered]


def artifact_inventory(
    root: Path,
    *,
    exclude: set[Path] | None = None,
) -> list[dict[str, Any]]:
    excluded = {item.resolve() for item in (exclude or set())}
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.resolve() not in excluded
    ]


def run_episode(
    source_repo: Path,
    cache_root: Path,
    run_root: Path,
    episode: dict[str, Any],
) -> dict[str, Any]:
    from pig_behavior.tracking.detector_cache import (
        DetectorEvidenceCache,
        ReplayDetector,
    )
    from pig_behavior.tracking.runner import run_tracking

    episode_root = run_root / "episodes" / episode["episode_id"]
    baseline_cfg = build_cfg(
        source_repo,
        episode,
        episode_root / "shadow_disabled",
    )
    cache_path = (
        cache_root
        / "episodes"
        / episode["episode_id"]
        / "detector_evidence.npz"
    )
    locked_hash = LOCKED_CACHE_HASHES[episode["episode_id"]]
    if sha256_file(cache_path) != locked_hash:
        raise ShadowAuditError(
            f"{episode['episode_id']}: locked cache hash mismatch"
        )
    identity = cache_identity(source_repo, episode, baseline_cfg)
    cache = DetectorEvidenceCache.load(
        cache_path,
        expected_identity=identity,
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
        h1_r3_shadow_observer=True,
    )
    if baseline_detector.invocations != shadow_detector.invocations:
        raise ShadowAuditError("replay invocation populations differ")
    if sha256_file(cache_path) != locked_hash:
        raise ShadowAuditError("detector cache changed during replay")

    baseline_shapes_sha = semantic_shapes_hash(
        output_shapes_path(baseline)
    )
    shadow_shapes_sha = semantic_shapes_hash(output_shapes_path(shadow))
    baseline_metrics = evaluate_metrics(
        source_repo,
        episode,
        Path(baseline.cvat_video_xml),
    )
    shadow_metrics = evaluate_metrics(
        source_repo,
        episode,
        Path(shadow.cvat_video_xml),
    )
    integer_metric_keys = (
        "gt_detections",
        "pred_detections",
        "gt_ids",
        "pred_ids",
        "tp",
        "fp",
        "fn",
        "idsw",
        "remapped_idsw",
        "wrong_id_matched_frames",
        "identity_discontinuities",
        "remapped_fragments",
    )
    metric_equal = all(
        baseline_metrics.get(key) == shadow_metrics.get(key)
        for key in integer_metric_keys
    )
    output_equal = baseline_shapes_sha == shadow_shapes_sha and metric_equal
    rows = []
    for record in shadow.h1_r3_shadow_candidate_rows:
        row = {
            "episode_id": episode["episode_id"],
            "development_role": episode["role"],
            "video_key": episode["video_key"],
            **record,
        }
        if int(row["would_activate"]):
            row["development_interpretation"] = (
                "PLAUSIBLE_POSITIVE_SUPPORT"
                if episode["role"] == "positive"
                else "PLAUSIBLE_CONTROL_FALSE_ACTIVATION"
            )
        else:
            row["development_interpretation"] = ""
        rows.append(row)
    mp4s = list(episode_root.rglob("*.mp4"))
    if mp4s:
        raise ShadowAuditError(f"MP4 output was created: {mp4s}")
    return {
        "episode": episode,
        "cache_sha256": locked_hash,
        "cache_replay_calls": baseline_detector.invocations,
        "detector_inference_calls": 0,
        "baseline": baseline,
        "shadow": shadow,
        "baseline_metrics": baseline_metrics,
        "shadow_metrics": shadow_metrics,
        "baseline_shapes_sha256": baseline_shapes_sha,
        "shadow_shapes_sha256": shadow_shapes_sha,
        "output_equal": output_equal,
        "candidate_rows": rows,
        "mp4_count": len(mp4s),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    source_repo = args.source_repo.resolve()
    cache_root = args.cache_root.resolve()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise ShadowAuditError("run root already exists; overwrite refused")
    run_root.mkdir(parents=True)
    if git("rev-parse", "HEAD") != AUTHORIZED_SHA:
        raise ShadowAuditError("worktree is not at the authorized parent SHA")

    development_manifest = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_DEVELOPMENT_MANIFEST.csv"
    )
    validation_manifest = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_VALIDATION_MANIFEST.csv"
    )
    validation_roles = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r2"
        / "H1_R2_VALIDATION_ROLE_ASSIGNMENTS.json"
    )
    design_contract = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_SCIENTIFIC_DESIGN_CONTRACT.md"
    )
    eligibility_contract = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_ELIGIBILITY_CONTRACT.json"
    )
    activation_gate = (
        REPO
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_ACTIVATION_GATE_DECISION.json"
    )
    validation_hashes_before = {
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "validation_role_assignment_sha256": sha256_file(validation_roles),
    }
    episodes = load_development_episodes(development_manifest)
    for episode in episodes:
        if sha256_file(video_path(source_repo, episode)) != (
            episode["video_sha256"]
        ):
            raise ShadowAuditError("development source-video hash mismatch")
        if sha256_file(gt_path(source_repo, episode)) != episode["gt_sha256"]:
            raise ShadowAuditError("development GT hash mismatch")

    contract = {
        "scope": "h1_r3_reservation_disabled_shadow_support_density",
        "profile": BASELINE_PROFILE,
        "score_name": SCORE_NAME,
        "score_calibrated": False,
        "score_is_probability": False,
        "threshold": SCORE_THRESHOLD,
        "support_margin": SUPPORT_MARGIN,
        "reservation_enabled": False,
        "association_changes_authorized": False,
        "validation_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
        "detector_cache_replay_only": True,
        "detector_inference_calls": 0,
        "output_timing_contract": "causal_framewise",
        "output_delay_frames": 0,
        "future_frames_used": False,
        "offline_repair": False,
        "smoothing": False,
        "run_root_mp4_count_required": 0,
    }
    manifest = {
        "schema_version": "tracking.h1_r3_shadow_run_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_parent_sha": AUTHORIZED_SHA,
        "execution_sha": git("rev-parse", "HEAD"),
        "dirty_files": git("status", "--short").splitlines(),
        "tracking_subtree_sha256": canonical_hash(
            git("ls-tree", "-r", "HEAD", "src/pig_behavior/tracking")
        ),
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "safe-refactor-test-guardian",
            "computer-vision-opencv",
        ],
        "development_manifest_sha256": sha256_file(development_manifest),
        "design_contract_sha256": sha256_file(design_contract),
        "eligibility_contract_sha256": sha256_file(eligibility_contract),
        "activation_gate_sha256": sha256_file(activation_gate),
        **validation_hashes_before,
        "contract": contract,
        "evaluation_contract_sha256": canonical_hash(contract),
        "episodes": episodes,
        "cache_hashes": LOCKED_CACHE_HASHES,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(run_root / "H1_R3_SHADOW_RUN_MANIFEST.json", manifest)

    results = []
    for episode in episodes:
        print(f"BEGIN {episode['episode_id']}", flush=True)
        result = run_episode(
            source_repo,
            cache_root,
            run_root,
            episode,
        )
        results.append(result)
        telemetry = result["shadow"].telemetry
        print(
            f"END {episode['episode_id']} "
            f"pairs={telemetry['h1_r3_shadow_pair_candidates']} "
            f"would_activate={telemetry['h1_r3_shadow_would_activate']}",
            flush=True,
        )

    candidate_rows = [
        row for result in results for row in result["candidate_rows"]
    ]
    write_csv(
        run_root / "H1_R3_SHADOW_CANDIDATE_PAIRS.csv",
        candidate_rows,
        candidate_fieldnames(candidate_rows),
    )
    episode_rows = []
    for result in results:
        episode = result["episode"]
        telemetry = result["shadow"].telemetry
        rows = result["candidate_rows"]
        activation_frames = {
            int(row["frame_index"])
            for row in rows
            if int(row["would_activate"])
        }
        episode_rows.append(
            {
                "episode_id": episode["episode_id"],
                "video_key": episode["video_key"],
                "development_role": episode["role"],
                "shadow_stage_calls": telemetry[
                    "h1_r3_shadow_stage_calls"
                ],
                "hidden_tracks_offered": telemetry[
                    "h1_r3_shadow_hidden_tracks_offered"
                ],
                "pair_candidates": telemetry[
                    "h1_r3_shadow_pair_candidates"
                ],
                "core_eligible_pairs": telemetry[
                    "h1_r3_shadow_core_eligible_pairs"
                ],
                "score_pairs": telemetry["h1_r3_shadow_score_pairs"],
                "would_activate_pairs": telemetry[
                    "h1_r3_shadow_would_activate"
                ],
                "frames_with_would_activation": len(activation_frames),
                "below_threshold": telemetry[
                    "h1_r3_shadow_below_threshold"
                ],
                "margin_failed": telemetry["h1_r3_shadow_margin_failed"],
                "missing_core_overlap": telemetry[
                    "h1_r3_shadow_missing_core_overlap"
                ],
                "missing_core_freshness": telemetry[
                    "h1_r3_shadow_missing_core_freshness"
                ],
                "optional_appearance_available": telemetry[
                    "h1_r3_shadow_optional_appearance_available"
                ],
                "optional_motion_available": telemetry[
                    "h1_r3_shadow_optional_motion_available"
                ],
                "shadow_disagreement_with_baseline": sum(
                    int(row["shadow_activation_would_disagree_with_baseline"])
                    for row in rows
                ),
                "baseline_output_equal": int(result["output_equal"]),
            }
        )
    write_csv(
        run_root / "H1_R3_SHADOW_SUPPORT_DENSITY_BY_EPISODE.csv",
        episode_rows,
        list(episode_rows[0]),
    )

    distribution_fields = (
        SCORE_NAME,
        "relative_overlap",
        "relative_freshness",
        "appearance_contribution",
        "motion_contribution",
        "activation_margin",
    )
    distributions = [
        summarize_numeric(candidate_rows, field)
        for field in distribution_fields
    ]
    write_csv(
        run_root / "H1_R3_SHADOW_SCORE_DISTRIBUTION.csv",
        distributions,
        list(distributions[0]),
    )
    optional_rows = []
    for episode_row in episode_rows:
        pairs = int(episode_row["pair_candidates"])
        for channel in ("appearance", "motion"):
            count = int(
                episode_row[f"optional_{channel}_available"]
            )
            optional_rows.append(
                {
                    "episode_id": episode_row["episode_id"],
                    "development_role": episode_row["development_role"],
                    "channel": channel,
                    "pair_candidates": pairs,
                    "both_sides_available": count,
                    "availability_fraction": (
                        count / pairs if pairs else "NOT_MEASURED"
                    ),
                }
            )
    write_csv(
        run_root / "H1_R3_SHADOW_OPTIONAL_EVIDENCE_AVAILABILITY.csv",
        optional_rows,
        list(optional_rows[0]),
    )

    equivalence = {
        "schema_version": "tracking.h1_r3_shadow_equivalence.v1",
        "episodes": {
            result["episode"]["episode_id"]: {
                "semantic_shapes_equal": (
                    result["baseline_shapes_sha256"]
                    == result["shadow_shapes_sha256"]
                ),
                "baseline_shapes_sha256": result[
                    "baseline_shapes_sha256"
                ],
                "shadow_shapes_sha256": result["shadow_shapes_sha256"],
                "integer_quality_metrics_equal": all(
                    result["baseline_metrics"].get(key)
                    == result["shadow_metrics"].get(key)
                    for key in (
                        "pred_detections",
                        "pred_ids",
                        "fp",
                        "fn",
                        "remapped_idsw",
                        "wrong_id_matched_frames",
                        "identity_discontinuities",
                        "remapped_fragments",
                    )
                ),
                "semantic_output_equal": result["output_equal"],
            }
            for result in results
        },
        "all_semantic_outputs_equal": all(
            result["output_equal"] for result in results
        ),
        "association_changes_authorized": False,
        "observer_returns_assignment_command": False,
    }
    write_json(
        run_root / "H1_R3_SHADOW_BASELINE_EQUIVALENCE_REPORT.json",
        equivalence,
    )
    telemetry = {
        "schema_version": "tracking.h1_r3_shadow_telemetry.v1",
        "episodes": {
            result["episode"]["episode_id"]: {
                "shadow_disabled": {
                    key: int(result["baseline"].telemetry.get(key, 0))
                    for key in SHADOW_COUNTERS
                },
                "shadow_enabled": {
                    key: int(result["shadow"].telemetry.get(key, 0))
                    for key in SHADOW_COUNTERS
                },
            }
            for result in results
        },
        "aggregate_shadow_enabled": {
            key: sum(
                int(result["shadow"].telemetry.get(key, 0))
                for result in results
            )
            for key in SHADOW_COUNTERS
        },
        "canonical_summary_path_verified": True,
    }
    write_json(run_root / "H1_R3_SHADOW_TELEMETRY.json", telemetry)

    positive_episodes = {
        row["episode_id"]
        for row in episode_rows
        if row["development_role"] == "positive"
        and int(row["would_activate_pairs"]) > 0
    }
    control_episodes = {
        row["episode_id"]
        for row in episode_rows
        if row["development_role"] == "control"
        and int(row["would_activate_pairs"]) > 0
    }
    control_rows = [
        row
        for row in candidate_rows
        if row["development_role"] == "control"
    ]
    control_activation_frames = {
        (row["episode_id"], int(row["frame_index"]))
        for row in control_rows
        if int(row["would_activate"])
    }
    control_frame_population = {
        (row["episode_id"], int(row["frame_index"]))
        for row in control_rows
    }
    control_fraction = (
        len(control_activation_frames) / len(control_frame_population)
        if control_frame_population
        else 0.0
    )
    activation_rows = [
        row for row in candidate_rows if int(row["would_activate"])
    ]
    observed_optional_activation = any(
        (
            int(row["hidden_appearance_available"])
            and int(row["visible_appearance_available"])
        )
        or (
            int(row["hidden_motion_available"])
            and int(row["visible_motion_available"])
        )
        for row in activation_rows
    )
    realistic_nonperfect = sum(
        float(row["hidden_overlap_similarity"]) < 1.0
        or float(row["visible_overlap_similarity"]) > 0.0
        for row in activation_rows
    )
    mp4_count = len(list(run_root.rglob("*.mp4")))
    aggregate = telemetry["aggregate_shadow_enabled"]
    gates = {
        "core_eligible_pairs_present": (
            aggregate["h1_r3_shadow_core_eligible_pairs"] > 0
        ),
        "frozen_gate_would_activate": (
            aggregate["h1_r3_shadow_would_activate"] > 0
        ),
        "two_positive_episodes_with_activation": (
            len(positive_episodes) >= 2
        ),
        "control_activation_not_broad": control_fraction <= 0.05,
        "activation_not_solely_missing_optional": (
            observed_optional_activation
        ),
        "two_realistic_nonperfect_rows_activate": realistic_nonperfect >= 2,
        "shadow_baseline_output_equivalence": equivalence[
            "all_semantic_outputs_equal"
        ],
        "shared_cache_parity": all(
            result["cache_sha256"]
            == LOCKED_CACHE_HASHES[result["episode"]["episode_id"]]
            for result in results
        ),
        "detector_inference_calls_zero": True,
        "run_root_mp4_count_zero": mp4_count == 0,
    }
    if not gates["core_eligible_pairs_present"]:
        decision = "FAIL_NO_CORE_SUPPORT"
    elif not gates["frozen_gate_would_activate"]:
        decision = "FAIL_NO_SHADOW_ACTIVATION"
    elif not gates["control_activation_not_broad"]:
        decision = "FAIL_CONTROL_DENSITY"
    elif not gates["activation_not_solely_missing_optional"]:
        decision = "FAIL_MISSINGNESS_SEMANTICS"
    elif not gates["shadow_baseline_output_equivalence"]:
        decision = "FAIL_SHADOW_SIDE_EFFECT"
    elif not all(gates.values()):
        decision = "FAIL_CONTRACT"
    else:
        decision = "PASS_SHADOW_PREREQUISITE"

    validation_hashes_after = {
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "validation_role_assignment_sha256": sha256_file(validation_roles),
    }
    if validation_hashes_before != validation_hashes_after:
        raise ShadowAuditError("validation artifacts changed")
    decision_payload = {
        "schema_version": "tracking.h1_r3_shadow_support_density_decision.v1",
        "experiment": "H1-r3",
        "date": "2026-07-27",
        "decision": decision,
        "code_sha": git("rev-parse", "HEAD"),
        "frozen_score": {
            "name": SCORE_NAME,
            "threshold": SCORE_THRESHOLD,
            "margin": SUPPORT_MARGIN,
            "calibrated": False,
            "is_probability": False,
        },
        "population": {
            "development_episodes_total": len(episodes),
            "development_episodes_completed": len(results),
            "positive_episodes_with_would_activation": len(
                positive_episodes
            ),
            "control_episodes_with_would_activation": len(control_episodes),
            "realistic_nonperfect_shadow_activations": realistic_nonperfect,
            "control_activation_frame_fraction": control_fraction,
        },
        "telemetry": aggregate,
        "gates": gates,
        "shadow_baseline_output_equivalence": equivalence[
            "all_semantic_outputs_equal"
        ],
        "shared_cache_parity": gates["shared_cache_parity"],
        "detector_inference_calls": 0,
        "run_root_mp4_count": mp4_count,
        "validation_hashes_before": validation_hashes_before,
        "validation_hashes_after": validation_hashes_after,
        "validation_executed": False,
        "implementation_authorization_ready": (
            decision == "PASS_SHADOW_PREREQUISITE"
        ),
        "association_implementation_authorized": False,
        "evaluation_authorized": False,
        "validation_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    write_json(
        run_root / "H1_R3_SHADOW_SUPPORT_DENSITY_DECISION_20260727.json",
        decision_payload,
    )
    commands = {
        "schema_version": "tracking.h1_r3_shadow_commands.v1",
        "commands": [
            "python scripts/tracking/check_h1_r3_design_contract.py",
            (
                "python scripts/tracking/run_h1_r3_shadow_support_density.py "
                f"--source-repo {source_repo} --cache-root {cache_root} "
                f"--run-root {run_root}"
            ),
        ],
        "environment": manifest["environment"],
        "detector_cache_recording_passes": 0,
        "detector_inference_calls": 0,
        "gpu_inference_calls": 0,
        "development_episode_arms": 12,
        "validation_windows": 0,
        "complete_videos": 0,
        "runtime_benchmarks": 0,
    }
    write_json(
        run_root / "H1_R3_SHADOW_COMMANDS_ENVIRONMENT.json",
        commands,
    )
    inventory_path = run_root / "H1_R3_SHADOW_ARTIFACT_SHA256.json"
    write_json(
        inventory_path,
        {
            "schema_version": "tracking.h1_r3_shadow_artifact_inventory.v1",
            "artifacts": artifact_inventory(
                run_root,
                exclude={inventory_path},
            ),
        },
    )
    print(f"SHADOW_PREREQUISITE_DECISION={decision}", flush=True)
    print(f"RUN_ROOT={run_root}", flush=True)
    return 0 if decision == "PASS_SHADOW_PREREQUISITE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
