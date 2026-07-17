from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pig_behavior.evaluation.tracking.repeatability as repeatability
from pig_behavior.evaluation.tracking.assets import TrackingPair
from pig_behavior.evaluation.tracking.lineage import (
    file_sha256,
    payload_sha256,
    write_artifact_manifest,
)
from pig_behavior.evaluation.tracking.repeatability import (
    TrackingRepeatabilityAuditConfig,
    audit_tracking_repeatability,
    write_tracking_repeatability_audit,
)

VIDEO_STEMS = ("pig_a", "pig_b")
SOURCE_COMMIT = "a" * 40


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_prediction_xml(
    path: Path,
    *,
    timestamp: str,
    identity: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "<annotations><meta><task><name>pig</name>"
            f"<created>{timestamp}</created>"
            f"<updated>{timestamp}</updated></task>"
            f"<dumped>{timestamp}</dumped></meta>"
            '<track id="0" label="Pig_1">'
            '<box frame="0" xtl="0" ytl="0" xbr="20" ybr="20" '
            'outside="0">'
            f'<attribute name="ID">{identity}</attribute>'
            '<attribute name="Hidden">No</attribute>'
            "</box></track></annotations>"
        ),
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _ensure_inputs(tmp_path: Path) -> tuple[Path, Path, list[TrackingPair]]:
    input_root = tmp_path / "inputs"
    input_root.mkdir(exist_ok=True)
    weights = input_root / "weights.pt"
    mask = input_root / "mask.png"
    weights.write_bytes(b"weights")
    mask.write_bytes(b"mask")
    pairs = []
    for stem in VIDEO_STEMS:
        video = input_root / f"{stem}.avi"
        gt_xml = input_root / f"{stem}.xml"
        video.write_bytes(b"video")
        gt_xml.write_text(
            f"<annotations><meta><task><name>{stem}</name></task></meta>"
            "</annotations>",
            encoding="utf-8",
        )
        pairs.append(
            TrackingPair(
                video_stem=stem,
                video_path=video,
                gt_xml=gt_xml,
            )
        )
    return weights, mask, pairs


def _build_run(
    tmp_path: Path,
    *,
    label: str,
    timestamp: str,
    identity: str = "ID_1",
    first_video_idsw: int = 0,
    frame_time_multiplier: float = 1.0,
    peak_memory_multiplier: float = 1.0,
) -> tuple[Path, Path]:
    weights, mask, input_pairs = _ensure_inputs(tmp_path)
    run_dir = tmp_path / f"{label}_eval"
    prediction_root = tmp_path / f"{label}_pred"
    run_dir.mkdir()
    pairs = []
    assets = []
    metrics = []
    runtime_rows = []
    for index, input_pair in enumerate(input_pairs):
        pred_xml = (
            prediction_root
            / "realtime"
            / input_pair.video_stem
            / "annotations_cvat_video_1_1.xml"
        )
        _write_prediction_xml(
            pred_xml,
            timestamp=timestamp,
            identity=identity,
        )
        quality_report = pred_xml.with_name("tracking_quality_report.json")
        quality_report.write_text(
            json.dumps(
                {
                    "telemetry": {
                        "declared_delay_frames": 0,
                        "output_timing_contract": "online_causal",
                    }
                }
            ),
            encoding="utf-8",
        )
        pair = TrackingPair(
            video_stem=input_pair.video_stem,
            video_path=input_pair.video_path,
            gt_xml=input_pair.gt_xml,
            pred_xml=pred_xml,
        )
        pairs.append(pair)
        assets.append(
            {
                "video_stem": pair.video_stem,
                "video_path": str(pair.video_path.resolve()),
                "gt_xml": str(pair.gt_xml.resolve()),
                "pred_xml": str(pred_xml.resolve()),
                "has_prediction": True,
            }
        )
        idsw = first_video_idsw if index == 0 else 0
        metrics.append(
            {
                "video_stem": pair.video_stem,
                "pred_xml": str(pred_xml.resolve()),
                "remapped_idsw": idsw,
                "remapped_hota_pct": 95.0 - index,
                "remapped_idf1_pct": 96.0 - index,
                "fp": 2 + index,
                "fn": 1 + index,
            }
        )
        runtime_rows.append(
            {
                "video_stem": pair.video_stem,
                "telemetry_available": True,
                "frames_processed": 10,
                "frame_time_ms_total": (1000 + index) * frame_time_multiplier,
                "frame_time_ms_p95": 110 + index,
                "peak_process_rss_bytes": int(
                    (1000 + index) * peak_memory_multiplier
                ),
                "peak_cuda_memory_allocated_bytes": int(
                    (2000 + index) * peak_memory_multiplier
                ),
                "peak_cuda_memory_reserved_bytes": int(
                    (3000 + index) * peak_memory_multiplier
                ),
            }
        )
    metrics.append(
        {
            "video_stem": "ALL",
            "pred_xml": "",
            "remapped_idsw": first_video_idsw,
            "remapped_hota_pct": 94.5,
            "remapped_idf1_pct": 95.5,
            "fp": 5,
            "fn": 3,
        }
    )
    _write_csv(run_dir / "tracking_eval_assets.csv", assets)
    _write_csv(run_dir / "tracking_metrics.csv", metrics)
    _write_csv(run_dir / "tracking_runtime_telemetry.csv", runtime_rows)
    (run_dir / "tracking_eval_config.json").write_text(
        json.dumps({"tracking_mode": "realtime"}),
        encoding="utf-8",
    )
    (run_dir / "tracking_report.md").write_text("# Tracking\n", encoding="utf-8")

    semantic_config = {
        "tracking_mode": "realtime",
        "profile_overrides": {"profile": "test"},
    }
    manifest = {
        "schema_version": 1,
        "status": "planned",
        "git": {"commit": SOURCE_COMMIT, "dirty": False, "dirty_entries": []},
        "selected_skills": ["tracking-experiment-guardian"],
        "config": {"prediction_root": str(prediction_root.resolve())},
        "semantic_config": semantic_config,
        "semantic_config_sha256": payload_sha256(semantic_config),
        "inputs": {
            "weights": _file_record(weights),
            "mask": _file_record(mask),
            "pairs": [
                {
                    "video_stem": pair.video_stem,
                    "video": _file_record(pair.video_path),
                    "gt_xml": _file_record(pair.gt_xml),
                    "prediction_xml": None,
                }
                for pair in pairs
            ],
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_artifact_manifest(run_dir, pairs)
    return run_dir, prediction_root


def _audit_config(primary: Path, repeated: Path) -> TrackingRepeatabilityAuditConfig:
    return TrackingRepeatabilityAuditConfig(
        primary_eval_dir=primary,
        repeat_eval_dir=repeated,
        expected_video_count=2,
        expected_commit=SOURCE_COMMIT,
        guard_video_max_remapped_idsw={"pig_a": 0},
        expected_delay_frames=0,
        expected_timing_contract="online_causal",
        require_clean_auditor=False,
    )


def test_repeatability_audit_passes_and_writes_fresh_lock(tmp_path: Path) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
    )
    config = _audit_config(primary, repeated)

    payload = audit_tracking_repeatability(config)

    assert payload["schema_version"] == 3
    assert payload["status"] == "PASS"
    assert payload["verified_prediction_count"] == 4
    assert payload["mp4_count"] == 0
    assert len(payload["prediction_semantic_hashes"]) == 2
    runtime_guardrails = payload["runtime"]["guardrails"]
    assert runtime_guardrails["tracking_loop_effective_fps"]["actual_ratio"] == 1.0
    assert runtime_guardrails["peak_memory"]["status"] == "PASS"
    output = tmp_path / "locks/repeatability.json"
    write_tracking_repeatability_audit(config, output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    with pytest.raises(FileExistsError, match="already exists"):
        write_tracking_repeatability_audit(config, output)


def test_repeatability_audit_rejects_semantic_prediction_drift(
    tmp_path: Path,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
        identity="ID_2",
    )

    with pytest.raises(ValueError, match="semantic predictions differ"):
        audit_tracking_repeatability(_audit_config(primary, repeated))


def test_repeatability_audit_rejects_metric_drift(tmp_path: Path) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
        first_video_idsw=1,
    )

    with pytest.raises(ValueError, match="metric mismatch"):
        audit_tracking_repeatability(_audit_config(primary, repeated))


def test_repeatability_audit_rejects_mp4_in_prediction_root(
    tmp_path: Path,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, repeat_prediction_root = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
    )
    forbidden = repeat_prediction_root / "stale.mp4"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"")

    with pytest.raises(RuntimeError, match="forbidden MP4"):
        audit_tracking_repeatability(_audit_config(primary, repeated))


def test_repeatability_audit_rejects_dirty_auditor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
    )
    monkeypatch.setattr(
        repeatability,
        "_auditor_git_state",
        lambda: {
            "project_root": "test",
            "commit": "b" * 40,
            "dirty": True,
            "dirty_entries": ["M checker.py"],
        },
    )
    config = replace(_audit_config(primary, repeated), require_clean_auditor=True)

    with pytest.raises(ValueError, match="auditor worktree must be clean"):
        audit_tracking_repeatability(config)


def test_repeatability_audit_rejects_runtime_fps_regression(
    tmp_path: Path,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
        frame_time_multiplier=2.0,
    )

    with pytest.raises(ValueError, match="Runtime FPS repeatability guard failed"):
        audit_tracking_repeatability(_audit_config(primary, repeated))


def test_repeatability_audit_rejects_peak_memory_regression(
    tmp_path: Path,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
        peak_memory_multiplier=1.25,
    )

    with pytest.raises(
        ValueError,
        match="Runtime memory repeatability guard failed",
    ):
        audit_tracking_repeatability(_audit_config(primary, repeated))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("min_repeat_tracking_fps_ratio", float("nan")),
        ("max_repeat_peak_memory_ratio", float("nan")),
        ("max_repeat_peak_memory_ratio", float("inf")),
    ],
)
def test_repeatability_audit_rejects_non_finite_runtime_guardrails(
    tmp_path: Path,
    field_name: str,
    invalid_value: float,
) -> None:
    primary, _ = _build_run(
        tmp_path,
        label="primary",
        timestamp="2026-07-17T01:00:00Z",
    )
    repeated, _ = _build_run(
        tmp_path,
        label="repeat",
        timestamp="2026-07-17T02:00:00Z",
    )
    config = replace(
        _audit_config(primary, repeated),
        **{field_name: invalid_value},
    )

    with pytest.raises(ValueError, match="must be finite"):
        audit_tracking_repeatability(config)
