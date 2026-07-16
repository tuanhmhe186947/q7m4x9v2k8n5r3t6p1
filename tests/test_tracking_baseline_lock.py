from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pig_behavior.evaluation.tracking.baseline_lock import (
    EXPECTED_MODE_METRICS,
    MODE_NAMES,
    lock_historical_baselines,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_xml(path: Path, task_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "<annotations>"
            f"<meta><task><name>{task_name}</name><size>1</size></task></meta>"
            "</annotations>"
        ),
        encoding="utf-8",
    )


def _build_eval_dir(
    *,
    eval_dir: Path,
    tracking_mode: str,
    prediction_xml: Path,
    video_path: Path,
    gt_xml: Path,
    weights_path: Path,
    mask_path: Path,
    metrics: dict[str, float | int],
) -> None:
    _write_xml(prediction_xml, video_path.stem)
    _write_csv(
        eval_dir / "tracking_eval_assets.csv",
        [
            {
                "video_stem": video_path.stem,
                "video_path": str(video_path),
                "gt_xml": str(gt_xml),
                "pred_xml": str(prediction_xml),
                "has_prediction": "True",
            }
        ],
    )
    metric_fields = {
        "remapped_idsw": 0,
        "remapped_hota_pct": 90.0,
        "remapped_idf1_pct": 90.0,
    }
    metric_fields.update(metrics)
    _write_csv(
        eval_dir / "tracking_metrics.csv",
        [
            {"video_stem": video_path.stem, **metric_fields},
            {"video_stem": "ALL", **metric_fields},
        ],
    )
    (eval_dir / "tracking_eval_config.json").write_text(
        json.dumps(
            {
                "tracking_mode": tracking_mode,
                "weights_path": str(weights_path),
                "mask_path": str(mask_path),
                "profile_overrides": {},
            }
        ),
        encoding="utf-8",
    )


def _build_baseline_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    video_path = data_dir / "pig_video.mp4"
    gt_xml = data_dir / "pig_video.xml"
    weights_path = tmp_path / "weights.pt"
    mask_path = tmp_path / "mask.png"
    data_dir.mkdir()
    video_path.write_bytes(b"video")
    _write_xml(gt_xml, video_path.stem)
    weights_path.write_bytes(b"weights")
    mask_path.write_bytes(b"mask")

    hybrid_dir = tmp_path / "hybrid"
    _build_eval_dir(
        eval_dir=hybrid_dir,
        tracking_mode="hybrid_bytetrack",
        prediction_xml=tmp_path / "pred/hybrid_history.xml",
        video_path=video_path,
        gt_xml=gt_xml,
        weights_path=weights_path,
        mask_path=mask_path,
        metrics=EXPECTED_MODE_METRICS["hybrid_bytetrack"],
    )
    mode_root = tmp_path / "modes"
    for mode in MODE_NAMES:
        tracking_mode = "realtime" if mode.startswith("realtime_") else mode
        _build_eval_dir(
            eval_dir=mode_root / mode / "iou0_area0_condarea0_merge0",
            tracking_mode=tracking_mode,
            prediction_xml=tmp_path / f"pred/{mode}.xml",
            video_path=video_path,
            gt_xml=gt_xml,
            weights_path=weights_path,
            mask_path=mask_path,
            metrics=EXPECTED_MODE_METRICS.get(mode, {}),
        )
    return hybrid_dir, mode_root


def test_lock_historical_baselines_hashes_same_universe(
    tmp_path: Path,
) -> None:
    hybrid_dir, mode_root = _build_baseline_tree(tmp_path)
    output_path = tmp_path / "locks/baseline.json"

    lock_historical_baselines(
        hybrid_eval_dir=hybrid_dir,
        mode_compare_root=mode_root,
        output_path=output_path,
        expected_video_count=1,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["source_commit"] is None
    assert payload["hybrid"]["video_count"] == 1
    assert set(payload["mode_comparison"]["modes"]) == set(MODE_NAMES)
    assert len(payload["universe_sha256"]) == 64


def test_lock_historical_baselines_rejects_mp4_in_output_tree(
    tmp_path: Path,
) -> None:
    hybrid_dir, mode_root = _build_baseline_tree(tmp_path)
    (mode_root / "forbidden.mp4").write_bytes(b"")

    with pytest.raises(RuntimeError, match="forbidden MP4"):
        lock_historical_baselines(
            hybrid_eval_dir=hybrid_dir,
            mode_compare_root=mode_root,
            output_path=tmp_path / "lock.json",
            expected_video_count=1,
        )


def test_lock_historical_baselines_can_record_incomplete_evidence(
    tmp_path: Path,
) -> None:
    hybrid_dir, mode_root = _build_baseline_tree(tmp_path)
    for prediction_xml in (tmp_path / "pred").glob("*.xml"):
        prediction_xml.unlink()
    output_path = tmp_path / "incomplete.json"

    with pytest.raises(FileNotFoundError, match="prediction XML is missing"):
        lock_historical_baselines(
            hybrid_eval_dir=hybrid_dir,
            mode_compare_root=mode_root,
            output_path=output_path,
            expected_video_count=1,
        )

    lock_historical_baselines(
        hybrid_eval_dir=hybrid_dir,
        mode_compare_root=mode_root,
        output_path=output_path,
        expected_video_count=1,
        allow_missing_predictions=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "INCOMPLETE"
    assert payload["hybrid"]["missing_prediction_count"] == 1
    assert payload["mode_comparison"]["modes"]["realtime_fast"][
        "missing_prediction_count"
    ] == 1
