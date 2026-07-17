"""Fail-closed primary/repeat audits for tracking experiments."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from .artifact_guard import assert_no_mp4_artifacts
from .assets import find_project_root
from .lineage import (
    CVAT_PREDICTION_SEMANTIC_HASH_CONTRACT,
    cvat_prediction_semantic_sha256,
    file_sha256,
    payload_sha256,
)

IGNORED_REPEAT_METRIC_FIELDS = frozenset({"pred_xml"})
REQUIRED_EVALUATION_ARTIFACTS = frozenset(
    {
        "run_manifest.json",
        "tracking_eval_assets.csv",
        "tracking_eval_config.json",
        "tracking_metrics.csv",
        "tracking_report.md",
        "tracking_runtime_telemetry.csv",
    }
)


@dataclass(frozen=True)
class TrackingRepeatabilityAuditConfig:
    """Inputs and hard guardrails for one primary/repeat audit."""

    primary_eval_dir: Path
    repeat_eval_dir: Path
    expected_video_count: int = 13
    expected_commit: str | None = None
    guard_video_max_remapped_idsw: dict[str, int] = field(default_factory=dict)
    expected_delay_frames: int | None = None
    expected_timing_contract: str | None = None
    min_repeat_tracking_fps_ratio: float = 0.90
    max_repeat_peak_memory_ratio: float = 1.10
    verify_input_hashes: bool = True
    require_clean_auditor: bool = True


def _auditor_git_state() -> dict[str, Any]:
    project_root = find_project_root(Path(__file__))

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Cannot bind audit tool git state: {result.stderr}")
        return result.stdout.strip()

    dirty_entries = run_git("status", "--short").splitlines()
    return {
        "project_root": str(project_root.resolve()),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required tracking JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Tracking JSON must contain an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required tracking CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_file_record(record: dict[str, Any], *, label: str) -> None:
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} size changed: {path}")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"{label} SHA256 changed: {path}")


def _validate_run_manifest(
    run_dir: Path,
    *,
    expected_commit: str | None,
    expected_video_count: int,
    verify_input_hashes: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = _read_json(run_dir / "run_manifest.json")
    git = manifest.get("git", {})
    if git.get("dirty") is not False:
        raise ValueError(f"Tracking run was not clean: {run_dir}")
    if expected_commit is not None and git.get("commit") != expected_commit:
        raise ValueError(
            "Tracking source commit mismatch: "
            f"expected={expected_commit}, actual={git.get('commit')}"
        )
    semantic_config = manifest.get("semantic_config")
    if not isinstance(semantic_config, dict):
        raise ValueError(f"Missing semantic config: {run_dir}")
    if payload_sha256(semantic_config) != manifest.get("semantic_config_sha256"):
        raise ValueError(f"Semantic config hash is invalid: {run_dir}")

    inputs = manifest.get("inputs", {})
    pairs = inputs.get("pairs", [])
    if len(pairs) != expected_video_count:
        raise ValueError(
            "Tracking input universe mismatch: "
            f"expected={expected_video_count}, actual={len(pairs)}"
        )
    input_stems = [str(pair["video_stem"]) for pair in pairs]
    if len(set(input_stems)) != expected_video_count:
        raise ValueError(f"Tracking input universe has duplicate stems: {run_dir}")
    if "tracking-experiment-guardian" not in manifest.get("selected_skills", []):
        raise ValueError(f"Tracking guardian skill is not recorded: {run_dir}")
    if verify_input_hashes:
        _assert_file_record(inputs["weights"], label="detector weights")
        if inputs.get("mask") is not None:
            _assert_file_record(inputs["mask"], label="tracking mask")
        for pair in pairs:
            stem = pair["video_stem"]
            _assert_file_record(pair["video"], label=f"video {stem}")
            _assert_file_record(pair["gt_xml"], label=f"GT XML {stem}")

    assets = _read_csv(run_dir / "tracking_eval_assets.csv")
    if len(assets) != expected_video_count:
        raise ValueError(
            "Tracking asset universe mismatch: "
            f"expected={expected_video_count}, actual={len(assets)}"
        )
    if any(row.get("has_prediction", "").lower() != "true" for row in assets):
        raise ValueError(f"Tracking asset row has no prediction: {run_dir}")
    asset_stems = [row["video_stem"] for row in assets]
    if len(set(asset_stems)) != expected_video_count:
        raise ValueError(f"Tracking assets have duplicate stems: {run_dir}")
    if set(asset_stems) != set(input_stems):
        raise ValueError(f"Tracking input and asset universes differ: {run_dir}")
    return manifest, assets


def _universe_contract(manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "video_stem": pair["video_stem"],
            "video_sha256": pair["video"]["sha256"],
            "gt_sha256": pair["gt_xml"]["sha256"],
        }
        for pair in manifest["inputs"]["pairs"]
    ]


def _prediction_root(
    manifest: dict[str, Any],
    assets: list[dict[str, str]],
) -> Path:
    parents = [str(Path(row["pred_xml"]).resolve().parent) for row in assets]
    fallback = Path(os.path.commonpath(parents))
    configured = manifest.get("config", {}).get("prediction_root")
    root = Path(configured).resolve() if configured else fallback
    if any(not Path(path).resolve().is_relative_to(root) for path in parents):
        raise ValueError(f"Prediction artifact escaped configured root: {root}")
    return root


def _validate_artifact_manifest(
    run_dir: Path,
    assets: list[dict[str, str]],
    *,
    expected_video_count: int,
) -> tuple[dict[str, Any], dict[str, str], dict[str, tuple[int, str]]]:
    path = run_dir / "artifact_manifest.json"
    manifest = _read_json(path)
    if manifest.get("mp4_count") != 0:
        raise ValueError(f"Artifact manifest does not assert zero MP4: {path}")
    expected_run_hash = file_sha256(run_dir / "run_manifest.json")
    if manifest.get("run_manifest_sha256") != expected_run_hash:
        raise ValueError(f"Run manifest hash mismatch: {path}")

    stem_by_prediction = {
        Path(row["pred_xml"]).resolve(): row["video_stem"] for row in assets
    }
    semantic_hashes: dict[str, str] = {}
    timing: dict[str, tuple[int, str]] = {}
    evaluation_names: set[str] = set()
    prediction_count = 0
    telemetry_count = 0
    for record in manifest.get("artifacts", []):
        artifact_path = Path(record["path"]).resolve()
        _assert_file_record(record, label=f"artifact {record.get('role')}")
        role = record.get("role")
        if role == "evaluation_output":
            if not artifact_path.is_relative_to(run_dir.resolve()):
                raise ValueError(f"Evaluation artifact escaped run root: {artifact_path}")
            evaluation_names.add(artifact_path.name)
        elif role == "prediction_xml":
            prediction_count += 1
            stem = stem_by_prediction.get(artifact_path)
            if stem is None:
                raise ValueError(f"Unexpected prediction artifact: {artifact_path}")
            contract = record.get("prediction_semantic_hash_contract")
            if contract != CVAT_PREDICTION_SEMANTIC_HASH_CONTRACT:
                raise ValueError(f"Prediction hash contract mismatch: {artifact_path}")
            semantic_hash = cvat_prediction_semantic_sha256(artifact_path)
            if semantic_hash != record.get("prediction_semantic_sha256"):
                raise ValueError(f"Prediction semantic hash changed: {artifact_path}")
            semantic_hashes[stem] = semantic_hash
        elif role == "prediction_runtime_telemetry":
            telemetry_count += 1
            stem = artifact_path.parent.name
            report = _read_json(artifact_path)
            telemetry = report.get("telemetry", {})
            timing[stem] = (
                int(telemetry["declared_delay_frames"]),
                str(telemetry["output_timing_contract"]),
            )

    if not REQUIRED_EVALUATION_ARTIFACTS.issubset(evaluation_names):
        missing = sorted(REQUIRED_EVALUATION_ARTIFACTS - evaluation_names)
        raise ValueError(f"Missing hashed evaluation artifacts: {missing}")
    expected_stems = {row["video_stem"] for row in assets}
    if (
        prediction_count != expected_video_count
        or set(semantic_hashes) != expected_stems
    ):
        raise ValueError(
            "Prediction artifact count mismatch: "
            f"expected={expected_video_count}, actual={prediction_count}"
        )
    if telemetry_count != expected_video_count or set(timing) != expected_stems:
        raise ValueError(
            "Runtime telemetry artifact count mismatch: "
            f"expected={expected_video_count}, actual={telemetry_count}"
        )
    return manifest, semantic_hashes, timing


def _metric_rows_by_stem(
    run_dir: Path,
    *,
    expected_video_count: int,
) -> dict[str, dict[str, str]]:
    rows = _read_csv(run_dir / "tracking_metrics.csv")
    expected_rows = expected_video_count + 1
    if len(rows) != expected_rows:
        raise ValueError(
            f"Metric row count mismatch: expected={expected_rows}, actual={len(rows)}"
        )
    by_stem = {row["video_stem"]: row for row in rows}
    if len(by_stem) != expected_rows or "ALL" not in by_stem:
        raise ValueError(f"Metric universe has duplicate or missing rows: {run_dir}")
    return by_stem


def _assert_metric_repeatability(
    primary: dict[str, dict[str, str]],
    repeated: dict[str, dict[str, str]],
) -> None:
    if set(primary) != set(repeated):
        raise ValueError("Primary/repeat metric video universes differ.")
    for stem, primary_row in primary.items():
        repeated_row = repeated[stem]
        if set(primary_row) != set(repeated_row):
            raise ValueError(f"Primary/repeat metric columns differ for {stem}.")
        for key, primary_value in primary_row.items():
            if key in IGNORED_REPEAT_METRIC_FIELDS:
                continue
            if primary_value != repeated_row[key]:
                raise ValueError(
                    f"Primary/repeat metric mismatch for {stem}.{key}: "
                    f"{primary_value!r} != {repeated_row[key]!r}"
                )


def _runtime_summary(run_dir: Path, *, expected_video_count: int) -> dict[str, Any]:
    rows = _read_csv(run_dir / "tracking_runtime_telemetry.csv")
    if len(rows) != expected_video_count:
        raise ValueError(f"Runtime telemetry row count mismatch: {run_dir}")
    if any(row.get("telemetry_available", "").lower() != "true" for row in rows):
        raise ValueError(f"Runtime telemetry is unavailable: {run_dir}")
    frames = sum(int(row["frames_processed"]) for row in rows)
    total_ms = sum(float(row["frame_time_ms_total"]) for row in rows)
    return {
        "frames_processed": frames,
        "tracking_loop_effective_fps": 1000.0 * frames / total_ms,
        "max_frame_time_ms_p95": max(
            float(row["frame_time_ms_p95"]) for row in rows
        ),
        "peak_process_rss_bytes": max(
            int(row["peak_process_rss_bytes"]) for row in rows
        ),
        "peak_cuda_memory_allocated_bytes": max(
            int(row["peak_cuda_memory_allocated_bytes"]) for row in rows
        ),
        "peak_cuda_memory_reserved_bytes": max(
            int(row["peak_cuda_memory_reserved_bytes"]) for row in rows
        ),
    }


def _runtime_repeatability_guardrails(
    primary: dict[str, Any],
    repeated: dict[str, Any],
    *,
    min_fps_ratio: float,
    max_memory_ratio: float,
) -> dict[str, Any]:
    if not isfinite(min_fps_ratio) or not 0.0 < min_fps_ratio <= 1.0:
        raise ValueError("Minimum repeat FPS ratio must be finite and in (0, 1].")
    if not isfinite(max_memory_ratio) or max_memory_ratio < 1.0:
        raise ValueError("Maximum repeat memory ratio must be finite and at least 1.")

    primary_fps = float(primary["tracking_loop_effective_fps"])
    repeat_fps = float(repeated["tracking_loop_effective_fps"])
    if (
        not isfinite(primary_fps)
        or primary_fps <= 0.0
        or not isfinite(repeat_fps)
        or repeat_fps <= 0.0
    ):
        raise ValueError("Primary and repeat effective FPS must be finite and positive.")
    fps_ratio = repeat_fps / primary_fps if primary_fps > 0.0 else 0.0
    if fps_ratio < min_fps_ratio:
        raise ValueError(
            "Runtime FPS repeatability guard failed: "
            f"minimum_ratio={min_fps_ratio:.3f}, actual_ratio={fps_ratio:.3f}, "
            f"primary_fps={primary_fps:.3f}, repeat_fps={repeat_fps:.3f}"
        )

    memory_results: dict[str, dict[str, float | int | str]] = {}
    memory_fields = (
        "peak_process_rss_bytes",
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    )
    for metric_name in memory_fields:
        primary_bytes = int(primary[metric_name])
        repeat_bytes = int(repeated[metric_name])
        if primary_bytes < 0 or repeat_bytes < 0:
            raise ValueError(f"Runtime memory telemetry must be non-negative: {metric_name}")
        if primary_bytes == 0:
            ratio = 1.0 if repeat_bytes == 0 else float("inf")
        else:
            ratio = repeat_bytes / primary_bytes
        if ratio > max_memory_ratio:
            raise ValueError(
                "Runtime memory repeatability guard failed: "
                f"field={metric_name}, maximum_ratio={max_memory_ratio:.3f}, "
                f"actual_ratio={ratio:.3f}, primary_bytes={primary_bytes}, "
                f"repeat_bytes={repeat_bytes}"
            )
        memory_results[metric_name] = {
            "primary_bytes": primary_bytes,
            "repeat_bytes": repeat_bytes,
            "ratio": ratio,
            "status": "PASS",
        }

    return {
        "tracking_loop_effective_fps": {
            "primary": primary_fps,
            "repeat": repeat_fps,
            "minimum_ratio": min_fps_ratio,
            "actual_ratio": fps_ratio,
            "status": "PASS",
        },
        "peak_memory": {
            "maximum_ratio": max_memory_ratio,
            "fields": memory_results,
            "status": "PASS",
        },
    }


def _weak_videos(rows: dict[str, dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    videos = [row for stem, row in rows.items() if stem != "ALL"]
    lowest_hota = sorted(videos, key=lambda row: float(row["remapped_hota_pct"]))
    highest_idsw = sorted(
        videos,
        key=lambda row: int(row["remapped_idsw"]),
        reverse=True,
    )
    return {
        "lowest_remapped_hota": [
            {
                "video_stem": row["video_stem"],
                "remapped_hota_pct": float(row["remapped_hota_pct"]),
            }
            for row in lowest_hota[:3]
        ],
        "highest_remapped_idsw": [
            {
                "video_stem": row["video_stem"],
                "remapped_idsw": int(row["remapped_idsw"]),
            }
            for row in highest_idsw[:3]
        ],
    }


def audit_tracking_repeatability(
    config: TrackingRepeatabilityAuditConfig,
) -> dict[str, Any]:
    """Validate two completed runs and return a hash-bound PASS record."""
    auditor = _auditor_git_state()
    if config.require_clean_auditor and auditor["dirty"]:
        raise ValueError(
            "Repeatability auditor worktree must be clean: "
            f"{auditor['dirty_entries']}"
        )
    primary_dir = config.primary_eval_dir.resolve()
    repeat_dir = config.repeat_eval_dir.resolve()
    if primary_dir == repeat_dir:
        raise ValueError("Primary and repeat evaluation roots must differ.")
    for run_dir in (primary_dir, repeat_dir):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Tracking evaluation root is missing: {run_dir}")
        assert_no_mp4_artifacts(run_dir, context="tracking repeatability audit")

    primary_run, primary_assets = _validate_run_manifest(
        primary_dir,
        expected_commit=config.expected_commit,
        expected_video_count=config.expected_video_count,
        verify_input_hashes=config.verify_input_hashes,
    )
    repeat_run, repeat_assets = _validate_run_manifest(
        repeat_dir,
        expected_commit=config.expected_commit,
        expected_video_count=config.expected_video_count,
        verify_input_hashes=config.verify_input_hashes,
    )
    if primary_run["git"]["commit"] != repeat_run["git"]["commit"]:
        raise ValueError("Primary/repeat source commits differ.")
    if primary_run["semantic_config_sha256"] != repeat_run["semantic_config_sha256"]:
        raise ValueError("Primary/repeat semantic configs differ.")
    if primary_run["inputs"]["weights"]["sha256"] != repeat_run["inputs"]["weights"]["sha256"]:
        raise ValueError("Primary/repeat detector hashes differ.")
    if primary_run["inputs"].get("mask") != repeat_run["inputs"].get("mask"):
        raise ValueError("Primary/repeat mask records differ.")

    primary_universe = _universe_contract(primary_run)
    repeat_universe = _universe_contract(repeat_run)
    if primary_universe != repeat_universe:
        raise ValueError("Primary/repeat video or GT universes differ.")
    primary_prediction_root = _prediction_root(primary_run, primary_assets)
    repeat_prediction_root = _prediction_root(repeat_run, repeat_assets)
    assert_no_mp4_artifacts(
        primary_prediction_root,
        context="primary tracking predictions",
    )
    assert_no_mp4_artifacts(
        repeat_prediction_root,
        context="repeat tracking predictions",
    )

    primary_artifacts, primary_hashes, primary_timing = (
        _validate_artifact_manifest(
            primary_dir,
            primary_assets,
            expected_video_count=config.expected_video_count,
        )
    )
    repeat_artifacts, repeat_hashes, repeat_timing = _validate_artifact_manifest(
        repeat_dir,
        repeat_assets,
        expected_video_count=config.expected_video_count,
    )
    if primary_hashes != repeat_hashes:
        changed = sorted(
            stem
            for stem in set(primary_hashes) | set(repeat_hashes)
            if primary_hashes.get(stem) != repeat_hashes.get(stem)
        )
        raise ValueError(f"Primary/repeat semantic predictions differ: {changed}")
    if primary_timing != repeat_timing:
        raise ValueError("Primary/repeat output timing contracts differ.")
    for stem, (delay_frames, timing_contract) in repeat_timing.items():
        if (
            config.expected_delay_frames is not None
            and delay_frames != config.expected_delay_frames
        ):
            raise ValueError(f"Unexpected declared delay for {stem}: {delay_frames}")
        if (
            config.expected_timing_contract is not None
            and timing_contract != config.expected_timing_contract
        ):
            raise ValueError(
                f"Unexpected output timing contract for {stem}: {timing_contract}"
            )

    primary_metrics = _metric_rows_by_stem(
        primary_dir,
        expected_video_count=config.expected_video_count,
    )
    repeat_metrics = _metric_rows_by_stem(
        repeat_dir,
        expected_video_count=config.expected_video_count,
    )
    _assert_metric_repeatability(primary_metrics, repeat_metrics)
    guardrail_results = {}
    for stem, maximum in config.guard_video_max_remapped_idsw.items():
        if stem not in repeat_metrics:
            raise ValueError(f"IDSW guard video is missing: {stem}")
        actual = int(repeat_metrics[stem]["remapped_idsw"])
        if actual > maximum:
            raise ValueError(
                f"IDSW guard failed for {stem}: maximum={maximum}, actual={actual}"
            )
        guardrail_results[stem] = {
            "maximum_remapped_idsw": maximum,
            "actual_remapped_idsw": actual,
            "status": "PASS",
        }

    primary_runtime = _runtime_summary(
        primary_dir,
        expected_video_count=config.expected_video_count,
    )
    repeat_runtime = _runtime_summary(
        repeat_dir,
        expected_video_count=config.expected_video_count,
    )
    runtime_guardrails = _runtime_repeatability_guardrails(
        primary_runtime,
        repeat_runtime,
        min_fps_ratio=config.min_repeat_tracking_fps_ratio,
        max_memory_ratio=config.max_repeat_peak_memory_ratio,
    )

    artifact_count = len(primary_artifacts["artifacts"]) + len(
        repeat_artifacts["artifacts"]
    )
    result = {
        "schema_version": 3,
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "auditor": auditor,
        "source_commit": primary_run["git"]["commit"],
        "semantic_config_sha256": primary_run["semantic_config_sha256"],
        "universe_sha256": payload_sha256(primary_universe),
        "prediction_semantic_hash_contract": (
            CVAT_PREDICTION_SEMANTIC_HASH_CONTRACT
        ),
        "prediction_semantic_hashes": primary_hashes,
        "metrics_equal_ignoring": sorted(IGNORED_REPEAT_METRIC_FIELDS),
        "aggregate_metrics": repeat_metrics["ALL"],
        "weak_videos": _weak_videos(repeat_metrics),
        "guardrails": guardrail_results,
        "timing_contracts": {
            stem: {
                "declared_delay_frames": timing[0],
                "output_timing_contract": timing[1],
            }
            for stem, timing in repeat_timing.items()
        },
        "runtime": {
            "primary": primary_runtime,
            "repeat": repeat_runtime,
            "guardrails": runtime_guardrails,
        },
        "primary": {
            "evaluation_dir": str(primary_dir),
            "prediction_root": str(primary_prediction_root),
            "run_manifest_sha256": file_sha256(primary_dir / "run_manifest.json"),
            "artifact_manifest_sha256": file_sha256(
                primary_dir / "artifact_manifest.json"
            ),
        },
        "repeat": {
            "evaluation_dir": str(repeat_dir),
            "prediction_root": str(repeat_prediction_root),
            "run_manifest_sha256": file_sha256(repeat_dir / "run_manifest.json"),
            "artifact_manifest_sha256": file_sha256(
                repeat_dir / "artifact_manifest.json"
            ),
        },
        "verified_artifact_count": artifact_count,
        "verified_prediction_count": 2 * config.expected_video_count,
        "mp4_count": 0,
    }
    result["authority_sha256"] = payload_sha256(result)
    return result


def write_tracking_repeatability_audit(
    config: TrackingRepeatabilityAuditConfig,
    output_path: Path,
) -> Path:
    """Audit two runs and write one immutable authority record."""
    resolved = output_path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Repeatability audit output already exists: {resolved}")
    result = audit_tracking_repeatability(config)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resolved


__all__ = [
    "TrackingRepeatabilityAuditConfig",
    "audit_tracking_repeatability",
    "write_tracking_repeatability_audit",
]
