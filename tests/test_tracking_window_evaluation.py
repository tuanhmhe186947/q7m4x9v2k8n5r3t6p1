from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from pig_behavior.evaluation.tracking.assets import TrackingPair
from pig_behavior.evaluation.tracking.cli import config_from_args, parse_args
from pig_behavior.evaluation.tracking.config import (
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml
from pig_behavior.evaluation.tracking.diagnostics import (
    continuity_gaps_for_pair,
    identity_events_for_pair,
    identity_mapping_for_pair,
)
from pig_behavior.evaluation.tracking.evaluator import evaluate_pair
from pig_behavior.evaluation.tracking.frame_window import (
    validate_generated_frame_coverage,
)
from pig_behavior.evaluation.tracking.lineage import prepare_run_manifest


def _write_xml(
    path: Path,
    rows: list[tuple[int, str, tuple[float, float, float, float]]],
) -> None:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "pig_video"
    ET.SubElement(task, "size").text = "5"
    ids = sorted({obj_id for _frame, obj_id, _bbox in rows})
    for track_index, obj_id in enumerate(ids):
        track = ET.SubElement(
            root,
            "track",
            {"id": str(track_index), "label": obj_id.replace("ID", "Pig")},
        )
        for frame, row_id, bbox in rows:
            if row_id != obj_id:
                continue
            xtl, ytl, xbr, ybr = bbox
            box = ET.SubElement(
                track,
                "box",
                {
                    "frame": str(frame),
                    "outside": "0",
                    "xtl": str(xtl),
                    "ytl": str(ytl),
                    "xbr": str(xbr),
                    "ybr": str(ybr),
                },
            )
            ET.SubElement(box, "attribute", {"name": "ID"}).text = obj_id
            ET.SubElement(box, "attribute", {"name": "Hidden"}).text = "No"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _make_pair(tmp_path: Path) -> TrackingPair:
    bbox = (0.0, 0.0, 20.0, 20.0)
    gt_xml = tmp_path / "pig_video.xml"
    pred_xml = tmp_path / "pig_video_prediction.xml"
    _write_xml(gt_xml, [(frame, "ID_1", bbox) for frame in range(5)])
    _write_xml(
        pred_xml,
        [
            (0, "ID_9", bbox),
            (1, "ID_9", bbox),
            (2, "ID_1", bbox),
            (3, "ID_1", bbox),
            (4, "ID_1", bbox),
        ],
    )
    video_path = tmp_path / "pig_video.mp4"
    video_path.write_bytes(b"video")
    return TrackingPair(
        video_stem="pig_video",
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )


def test_parse_cvat_xml_uses_inclusive_frame_bounds(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)

    parsed = parse_cvat_video_xml(
        pair.gt_xml,
        start_frame=2,
        end_frame=4,
    )

    assert list(parsed) == [2, 3, 4]


def test_score_window_excludes_warmup_from_all_evidence(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)

    metrics = evaluate_pair(
        pair,
        evaluation_start_frame=2,
        evaluation_end_frame=4,
    )
    events = identity_events_for_pair(
        pair,
        evaluation_start_frame=2,
        evaluation_end_frame=4,
    )
    mapping = identity_mapping_for_pair(
        pair,
        evaluation_start_frame=2,
        evaluation_end_frame=4,
    )
    gaps = continuity_gaps_for_pair(
        pair,
        evaluation_start_frame=2,
        evaluation_end_frame=4,
    )

    assert metrics is not None
    assert metrics.evaluated_frames == 3
    assert metrics.gt_detections == 3
    assert metrics.pred_detections == 3
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.idsw == 0
    assert events == []
    assert {row["pred_id"] for row in mapping} == {"ID_1"}
    assert gaps == []


def test_absent_score_bounds_preserve_full_video_behavior(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)

    default_metrics = evaluate_pair(pair)
    explicit_metrics = evaluate_pair(
        pair,
        evaluation_start_frame=None,
        evaluation_end_frame=None,
    )

    assert default_metrics is not None
    assert explicit_metrics is not None
    assert asdict(default_metrics) == asdict(explicit_metrics)


@pytest.mark.parametrize(
    ("start_frame", "end_frame"),
    [(-1, 4), (5, 4)],
)
def test_invalid_score_bounds_fail_closed(
    start_frame: int,
    end_frame: int,
) -> None:
    with pytest.raises(ValueError, match="evaluation_"):
        TrackingEvaluationPipelineConfig(
            evaluation_start_frame=start_frame,
            evaluation_end_frame=end_frame,
        )


def test_generated_prediction_must_cover_score_window() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        validate_generated_frame_coverage(
            tracking_start_frame=100,
            max_frames=25,
            evaluation_start_frame=110,
            evaluation_end_frame=130,
        )

    with pytest.raises(ValueError, match="start at or before"):
        validate_generated_frame_coverage(
            tracking_start_frame=120,
            max_frames=25,
            evaluation_start_frame=110,
            evaluation_end_frame=130,
        )


def test_cli_propagates_score_window_to_pipeline_config(tmp_path: Path) -> None:
    video_path = tmp_path / "pig_video.mp4"
    video_path.write_bytes(b"video")
    args = parse_args(
        [
            "--video",
            str(video_path),
            "--evaluation-start-frame",
            "110",
            "--evaluation-end-frame",
            "130",
        ]
    )

    config = config_from_args(args)

    assert config.evaluation_start_frame == 110
    assert config.evaluation_end_frame == 130


def test_run_manifest_hash_binds_score_window(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)
    weights_path = tmp_path / "weights.pt"
    mask_path = tmp_path / "mask.png"
    weights_path.write_bytes(b"weights")
    mask_path.write_bytes(b"mask")
    config = TrackingEvaluationPipelineConfig(
        prediction_root=tmp_path / "predictions",
        output_root=tmp_path / "reports",
        weights_path=weights_path,
        mask_path=mask_path,
        run_missing_tracker=False,
        expected_video_count=1,
        evaluation_start_frame=2,
        evaluation_end_frame=4,
    )

    manifest_path = prepare_run_manifest([pair], config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["semantic_config"]["evaluation_start_frame"] == 2
    assert manifest["semantic_config"]["evaluation_end_frame"] == 4


def test_manifest_rejects_uncovered_generated_window(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)
    config = TrackingEvaluationPipelineConfig(
        prediction_root=tmp_path / "predictions",
        output_root=tmp_path / "reports",
        force_track=True,
        max_frames=25,
        evaluation_start_frame=110,
        evaluation_end_frame=130,
        profile_overrides={"start_frame": 100},
    )

    with pytest.raises(ValueError, match="does not cover"):
        prepare_run_manifest([pair], config)

    assert not config.output_root.exists()
