"""Validate and hash-lock historical tracking baseline evidence."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_guard import assert_no_mp4_artifacts
from .assets import TrackingPair, find_project_root
from .lineage import (
    LOCKED_BASELINES,
    SELECTED_TRACKING_SKILLS,
    file_sha256,
    payload_sha256,
    validate_tracking_pairs,
)

MODE_NAMES = (
    "bytetrack_raw",
    "hybrid_bytetrack",
    "realtime_fast",
    "realtime_balanced",
    "realtime_quality_delayed",
)
EXPECTED_MODE_METRICS = {
    "hybrid_bytetrack": {
        "remapped_idsw": 0,
        "remapped_hota_pct": 97.26,
        "remapped_idf1_pct": 98.58,
    },
    "realtime_fast": {
        "remapped_idsw": 56,
        "remapped_hota_pct": 93.10,
        "remapped_idf1_pct": 92.91,
    },
    "realtime_balanced": {
        "remapped_idsw": 75,
        "remapped_hota_pct": 92.77,
        "remapped_idf1_pct": 93.12,
    },
    "realtime_quality_delayed": {
        "remapped_idsw": 21,
        "remapped_hota_pct": 96.60,
        "remapped_idf1_pct": 97.02,
    },
}
PATH_CONFIG_FIELDS = {
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
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Baseline artifact not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _metric_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid baseline metric {key!r}: {row.get(key)!r}") from exc


def _assert_expected_metrics(
    aggregate: dict[str, str],
    expected: dict[str, float | int],
) -> None:
    for key, expected_value in expected.items():
        actual = _metric_value(aggregate, key)
        if abs(actual - float(expected_value)) > 0.005:
            raise ValueError(
                f"Baseline metric mismatch for {key}: "
                f"expected={expected_value}, actual={actual}"
            )


def _semantic_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config_payload.items()
        if key not in PATH_CONFIG_FIELDS
    }


def validate_evaluation_directory(
    eval_dir: Path,
    *,
    expected_mode: str,
    expected_video_count: int,
    expected_metrics: dict[str, float | int] | None = None,
    allow_missing_predictions: bool = False,
) -> dict[str, Any]:
    """Validate one historical evaluation directory and hash its evidence."""
    resolved_dir = eval_dir.resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Baseline evaluation directory missing: {eval_dir}")
    assert_no_mp4_artifacts(
        resolved_dir,
        context=f"historical baseline {expected_mode}",
    )

    assets_path = resolved_dir / "tracking_eval_assets.csv"
    metrics_path = resolved_dir / "tracking_metrics.csv"
    config_path = resolved_dir / "tracking_eval_config.json"
    assets = _read_csv(assets_path)
    metrics = _read_csv(metrics_path)
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_payload.get("tracking_mode") != expected_mode:
        raise ValueError(
            "Historical tracking mode mismatch: "
            f"expected={expected_mode}, "
            f"actual={config_payload.get('tracking_mode')}"
        )

    pairs = [
        TrackingPair(
            video_stem=row["video_stem"],
            video_path=Path(row["video_path"]),
            gt_xml=Path(row["gt_xml"]),
            pred_xml=Path(row["pred_xml"]),
        )
        for row in assets
    ]
    if any(str(row.get("has_prediction", "")).lower() != "true" for row in assets):
        raise ValueError("Historical baseline has missing prediction rows.")
    validate_tracking_pairs(
        pairs,
        expected_video_count=expected_video_count,
    )
    missing_predictions = [
        str(pair.pred_xml)
        for pair in pairs
        if pair.pred_xml is None or not pair.pred_xml.is_file()
    ]
    if missing_predictions and not allow_missing_predictions:
        raise FileNotFoundError(
            "Historical baseline prediction XML is missing: "
            f"{missing_predictions[:3]}"
        )

    metric_stems = [row.get("video_stem", "") for row in metrics]
    expected_stems = {pair.video_stem for pair in pairs}
    actual_stems = {stem for stem in metric_stems if stem != "ALL"}
    if actual_stems != expected_stems or metric_stems.count("ALL") != 1:
        raise ValueError(
            "Historical metric universe mismatch: "
            f"expected={sorted(expected_stems)}, actual={sorted(actual_stems)}"
        )
    aggregate = next(row for row in metrics if row.get("video_stem") == "ALL")
    if expected_metrics is not None:
        _assert_expected_metrics(aggregate, expected_metrics)

    weights_path = Path(config_payload["weights_path"])
    mask_value = config_payload.get("mask_path")
    mask_path = Path(mask_value) if mask_value else None
    input_rows = []
    for pair in pairs:
        prediction_record = None
        if pair.pred_xml is not None and pair.pred_xml.is_file():
            prediction_record = _file_record(pair.pred_xml)
        input_rows.append(
            {
            "video_stem": pair.video_stem,
            "video": _file_record(pair.video_path),
            "gt_xml": _file_record(pair.gt_xml),
                "prediction_xml": prediction_record,
                "recorded_prediction_path": (
                    str(pair.pred_xml.resolve())
                    if pair.pred_xml is not None
                    else None
                ),
            }
        )
    universe_contract = [
        {
            "video_stem": row["video_stem"],
            "video_sha256": row["video"]["sha256"],
            "gt_sha256": row["gt_xml"]["sha256"],
        }
        for row in input_rows
    ]
    evidence_files = [
        _file_record(path)
        for path in sorted(resolved_dir.rglob("*"))
        if path.is_file()
    ]
    semantic_config = _semantic_config(config_payload)
    return {
        "status": "INCOMPLETE" if missing_predictions else "PASS",
        "mode": expected_mode,
        "evaluation_dir": str(resolved_dir),
        "video_count": len(pairs),
        "universe_sha256": payload_sha256(universe_contract),
        "semantic_config": semantic_config,
        "semantic_config_sha256": payload_sha256(semantic_config),
        "weights": _file_record(weights_path),
        "mask": _file_record(mask_path) if mask_path is not None else None,
        "inputs": input_rows,
        "missing_prediction_count": len(missing_predictions),
        "missing_prediction_paths": missing_predictions,
        "aggregate_metrics": aggregate,
        "evidence_files": evidence_files,
        "mp4_count": 0,
    }


def lock_historical_baselines(
    *,
    hybrid_eval_dir: Path,
    mode_compare_root: Path,
    output_path: Path,
    expected_video_count: int = 13,
    allow_missing_predictions: bool = False,
) -> Path:
    """Write a fail-closed lock manifest for hybrid and five-mode evidence."""
    if output_path.exists():
        raise FileExistsError(f"Baseline lock output already exists: {output_path}")
    assert_no_mp4_artifacts(
        mode_compare_root,
        context="historical mode comparison",
    )
    hybrid = validate_evaluation_directory(
        hybrid_eval_dir,
        expected_mode="hybrid_bytetrack",
        expected_video_count=expected_video_count,
        expected_metrics=EXPECTED_MODE_METRICS["hybrid_bytetrack"],
        allow_missing_predictions=allow_missing_predictions,
    )
    modes = {
        mode: validate_evaluation_directory(
            mode_compare_root / mode / "iou0_area0_condarea0_merge0",
            expected_mode=(
                "realtime" if mode.startswith("realtime_") else mode
            ),
            expected_video_count=expected_video_count,
            expected_metrics=EXPECTED_MODE_METRICS.get(mode),
            allow_missing_predictions=allow_missing_predictions,
        )
        for mode in MODE_NAMES
    }
    universe_hashes = {
        hybrid["universe_sha256"],
        *(record["universe_sha256"] for record in modes.values()),
    }
    if len(universe_hashes) != 1:
        raise ValueError(f"Baseline universe hashes differ: {universe_hashes}")

    project_root = find_project_root(Path(__file__))
    records = [hybrid, *modes.values()]
    incomplete = any(record["status"] != "PASS" for record in records)
    payload = {
        "schema_version": 1,
        "status": "INCOMPLETE" if incomplete else "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generator_commit": _git_commit(project_root),
        "source_commit": None,
        "source_commit_status": "unavailable_for_historical_runs",
        "selected_skills": list(SELECTED_TRACKING_SKILLS),
        "locked_baselines": LOCKED_BASELINES,
        "universe_sha256": next(iter(universe_hashes)),
        "hybrid": hybrid,
        "mode_comparison": {
            "root": str(mode_compare_root.resolve()),
            "modes": modes,
            "mp4_count": 0,
        },
        "limitations": [
            "Historical runs predate commit-bound run manifests.",
            "Weight hashes bind the current files at their recorded paths.",
            *(
                [
                    "Historical prediction XML roots are missing; metrics and "
                    "reports are locked, but prediction bytes are not."
                ]
                if incomplete
                else []
            ),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "EXPECTED_MODE_METRICS",
    "MODE_NAMES",
    "lock_historical_baselines",
    "validate_evaluation_directory",
]
