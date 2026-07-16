from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.evaluation.tracking.assets import TrackingPair
from pig_behavior.evaluation.tracking.config import (
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.lineage import (
    file_sha256,
    prepare_run_manifest,
    validate_metric_universe,
    validate_tracking_pairs,
)


def _write_cvat_xml(path: Path, task_name: str) -> None:
    path.write_text(
        (
            "<annotations>"
            f"<meta><task><name>{task_name}</name><size>1</size></task></meta>"
            '<track id="0" label="Pig_1">'
            '<box frame="0" xtl="0" ytl="0" xbr="20" ybr="20" '
            'outside="0">'
            '<attribute name="ID">ID_1</attribute>'
            '<attribute name="Hidden">No</attribute>'
            "</box></track></annotations>"
        ),
        encoding="utf-8",
    )


def _make_pair(tmp_path: Path, stem: str = "pig_video") -> TrackingPair:
    video_path = tmp_path / f"{stem}.mp4"
    gt_xml = tmp_path / f"{stem}.xml"
    pred_xml = tmp_path / f"{stem}_prediction.xml"
    video_path.write_bytes(b"video")
    _write_cvat_xml(gt_xml, stem)
    _write_cvat_xml(pred_xml, stem)
    return TrackingPair(
        video_stem=stem,
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )


def _make_config(
    tmp_path: Path,
    *,
    force_track: bool = False,
) -> TrackingEvaluationPipelineConfig:
    weights_path = tmp_path / "weights.pt"
    mask_path = tmp_path / "mask.png"
    weights_path.write_bytes(b"weights")
    mask_path.write_bytes(b"mask")
    return TrackingEvaluationPipelineConfig(
        prediction_root=tmp_path / "prediction_root",
        output_root=tmp_path / "evaluation_root",
        weights_path=weights_path,
        mask_path=mask_path,
        run_missing_tracker=False,
        force_track=force_track,
        expected_video_count=1,
        tracking_mode="hybrid_bytetrack",
        profile_overrides={"det_conf": 0.2},
    )


def test_prepare_run_manifest_hash_binds_inputs_and_skills(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)
    config = _make_config(tmp_path)

    manifest_path = prepare_run_manifest([pair], config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["status"] == "planned"
    assert payload["inputs"]["weights"]["sha256"] == file_sha256(
        config.weights_path
    )
    assert payload["inputs"]["pairs"][0]["video"]["sha256"] == file_sha256(
        pair.video_path
    )
    assert payload["inputs"]["pairs"][0]["gt_xml"]["sha256"] == file_sha256(
        pair.gt_xml
    )
    assert "tracking-experiment-guardian" in payload["selected_skills"]
    assert payload["semantic_config"]["tracking_mode"] == "hybrid_bytetrack"
    assert len(payload["semantic_config_sha256"]) == 64


def test_prepare_run_manifest_rejects_reused_output_root(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)
    config = _make_config(tmp_path)
    config.output_root.mkdir()

    with pytest.raises(FileExistsError, match="output root already exists"):
        prepare_run_manifest([pair], config)


def test_prepare_run_manifest_rejects_reused_prediction_root_for_tracking(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)
    pair.pred_xml = None
    config = _make_config(tmp_path, force_track=True)
    config.prediction_root.mkdir()

    with pytest.raises(FileExistsError, match="Prediction output root"):
        prepare_run_manifest([pair], config)


def test_validate_tracking_pairs_rejects_duplicate_stem(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)

    with pytest.raises(ValueError, match="Duplicate tracking video stems"):
        validate_tracking_pairs(
            [pair, pair],
            expected_video_count=2,
        )


def test_validate_tracking_pairs_rejects_wrong_gt_mapping(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)
    wrong_gt = tmp_path / "unrelated.xml"
    _write_cvat_xml(wrong_gt, "different_task")
    pair.gt_xml = wrong_gt

    with pytest.raises(ValueError, match="GT stem/task mismatch"):
        validate_tracking_pairs([pair], expected_video_count=1)


def test_validate_tracking_pairs_rejects_incomplete_universe(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)

    with pytest.raises(ValueError, match="expected 13, found 1"):
        validate_tracking_pairs([pair], expected_video_count=13)


def test_validate_metric_universe_requires_exact_rows(
    tmp_path: Path,
) -> None:
    pair = _make_pair(tmp_path)
    valid = pd.DataFrame(
        [
            {"video_stem": pair.video_stem},
            {"video_stem": "ALL"},
        ]
    )
    validate_metric_universe(valid, [pair])

    missing = pd.DataFrame([{"video_stem": "ALL"}])
    with pytest.raises(ValueError, match="metric universe mismatch"):
        validate_metric_universe(missing, [pair])
