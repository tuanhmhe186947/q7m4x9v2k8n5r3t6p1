"""Hash-bound lineage gates for tracking evaluation runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_guard import assert_no_mp4_artifacts
from .assets import (
    TrackingPair,
    find_project_root,
    normalize_key,
    resolve_mask_path,
)
from .config import TrackingEvaluationPipelineConfig
from .cvat_io import read_task_name

SELECTED_TRACKING_SKILLS = (
    "tracking-experiment-guardian",
    "computer-vision-opencv",
    "safe-refactor-test-guardian",
    "scientific-ablation-controller",
    "experiment-lineage-reproducibility",
)
LOCKED_BASELINES = {
    "hybrid": "20260707_230230",
    "mode_comparison": "20260709_040751",
}
NON_SEMANTIC_CONFIG_FIELDS = {
    "video_path",
    "video_paths",
    "gt_xml",
    "gt_dir",
    "video_dir",
    "prediction_root",
    "output_root",
    "weights_path",
    "weights_v26_path",
    "mask_path",
    "expected_video_count",
}


@lru_cache(maxsize=256)
def _sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one existing file while caching immutable inputs within a process."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def payload_sha256(payload: Any) -> str:
    """Hash a JSON-compatible semantic payload deterministically."""
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"

    dirty_entries = run_git("status", "--short").splitlines()
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def validate_tracking_pairs(
    pairs: list[TrackingPair],
    *,
    expected_video_count: int | None,
) -> None:
    """Reject missing, duplicate, or mismatched video/GT mappings."""
    if not pairs:
        raise ValueError("Tracking lineage has no video/GT pairs.")
    if expected_video_count is not None and len(pairs) != expected_video_count:
        raise ValueError(
            "Tracking universe mismatch: "
            f"expected {expected_video_count}, found {len(pairs)}."
        )

    stems = [pair.video_stem for pair in pairs]
    duplicate_stems = sorted(
        stem for stem, count in Counter(stems).items() if count > 1
    )
    if duplicate_stems:
        raise ValueError(f"Duplicate tracking video stems: {duplicate_stems}")

    video_paths: set[Path] = set()
    gt_paths: set[Path] = set()
    for pair in pairs:
        if not pair.video_path.is_file():
            raise FileNotFoundError(f"Tracking video not found: {pair.video_path}")
        if not pair.gt_xml.is_file():
            raise FileNotFoundError(f"Tracking GT not found: {pair.gt_xml}")
        if pair.video_path.stem != pair.video_stem:
            raise ValueError(
                f"Video stem mismatch: {pair.video_stem} != "
                f"{pair.video_path.stem}"
            )

        video_key = normalize_key(pair.video_stem)
        gt_name_key = normalize_key(pair.gt_xml.stem)
        gt_task_key = normalize_key(read_task_name(pair.gt_xml))
        if video_key not in gt_name_key and video_key not in gt_task_key:
            raise ValueError(
                f"GT stem/task mismatch for {pair.video_stem}: {pair.gt_xml}"
            )

        resolved_video = pair.video_path.resolve()
        resolved_gt = pair.gt_xml.resolve()
        if resolved_video in video_paths:
            raise ValueError(f"Duplicate tracking video path: {resolved_video}")
        if resolved_gt in gt_paths:
            raise ValueError(f"Duplicate tracking GT path: {resolved_gt}")
        video_paths.add(resolved_video)
        gt_paths.add(resolved_gt)


def _input_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def prepare_run_manifest(
    pairs: list[TrackingPair],
    config: TrackingEvaluationPipelineConfig,
) -> Path:
    """Validate fresh roots and write a planned manifest before tracking."""
    if config.output_root.exists():
        raise FileExistsError(
            f"Evaluation output root already exists: {config.output_root}"
        )
    should_track = any(
        config.force_track
        or (config.run_missing_tracker and pair.pred_xml is None)
        for pair in pairs
    )
    if should_track and config.prediction_root.exists():
        raise FileExistsError(
            f"Prediction output root already exists: {config.prediction_root}"
        )

    validate_tracking_pairs(
        pairs,
        expected_video_count=config.expected_video_count,
    )
    if not config.weights_path.is_file():
        raise FileNotFoundError(
            f"Tracking detector weights not found: {config.weights_path}"
        )
    mask_path = config.mask_path or resolve_mask_path()
    if mask_path is not None and not mask_path.is_file():
        raise FileNotFoundError(f"Tracking mask not found: {mask_path}")

    if config.prediction_root.exists():
        assert_no_mp4_artifacts(
            config.prediction_root,
            context="tracking prediction input",
        )

    config_payload = _jsonable(asdict(config))
    semantic_config = {
        key: value
        for key, value in config_payload.items()
        if key not in NON_SEMANTIC_CONFIG_FIELDS
    }
    manifest = {
        "schema_version": 1,
        "status": "planned",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(find_project_root(Path(__file__))),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd().resolve()),
        "selected_skills": list(SELECTED_TRACKING_SKILLS),
        "locked_baselines": LOCKED_BASELINES,
        "experiment_seed": None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "config": config_payload,
        "semantic_config": semantic_config,
        "semantic_config_sha256": payload_sha256(semantic_config),
        "inputs": {
            "weights": _input_record(config.weights_path),
            "mask": _input_record(mask_path) if mask_path is not None else None,
            "pairs": [
                {
                    "video_stem": pair.video_stem,
                    "video": _input_record(pair.video_path),
                    "gt_xml": _input_record(pair.gt_xml),
                    "prediction_xml": (
                        _input_record(pair.pred_xml)
                        if pair.pred_xml is not None and pair.pred_xml.is_file()
                        else None
                    ),
                }
                for pair in pairs
            ],
        },
    }
    config.output_root.mkdir(parents=True)
    manifest_path = config.output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def validate_metric_universe(
    metrics_df: pd.DataFrame,
    pairs: list[TrackingPair],
) -> None:
    """Require one per-video metric row plus one aggregate row."""
    expected = {
        pair.video_stem
        for pair in pairs
        if pair.pred_xml is not None and pair.pred_xml.is_file()
    }
    if "video_stem" not in metrics_df.columns:
        raise ValueError("Tracking metrics are missing video_stem.")
    stems = [str(value) for value in metrics_df["video_stem"].tolist()]
    actual = {stem for stem in stems if stem != "ALL"}
    if actual != expected:
        raise ValueError(
            "Tracking metric universe mismatch: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    counts = Counter(stems)
    duplicates = sorted(
        stem for stem, count in counts.items() if stem != "ALL" and count != 1
    )
    if duplicates or counts.get("ALL", 0) != 1:
        raise ValueError(
            "Tracking metrics require exactly one row per video and one ALL row."
        )


def _artifact_records(paths: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    records = []
    seen: set[Path] = set()
    for role, path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        record = {"role": role, **_input_record(resolved)}
        records.append(record)
    return sorted(records, key=lambda item: (item["role"], item["path"]))


def write_artifact_manifest(
    run_dir: Path,
    pairs: list[TrackingPair],
) -> Path:
    """Hash required reports and prediction XML after a successful run."""
    required_names = {
        "run_manifest.json",
        "tracking_eval_assets.csv",
        "tracking_eval_config.json",
        "tracking_metrics.csv",
        "tracking_report.md",
    }
    missing = sorted(
        name for name in required_names if not (run_dir / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Missing tracking run artifacts: {missing}")
    assert_no_mp4_artifacts(run_dir, context="tracking evaluation report")

    output_paths = [
        ("evaluation_output", path)
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    prediction_paths = [
        ("prediction_xml", pair.pred_xml)
        for pair in pairs
        if pair.pred_xml is not None and pair.pred_xml.is_file()
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
        "artifacts": _artifact_records([*output_paths, *prediction_paths]),
        "mp4_count": 0,
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "file_sha256",
    "payload_sha256",
    "prepare_run_manifest",
    "validate_metric_universe",
    "validate_tracking_pairs",
    "write_artifact_manifest",
]
