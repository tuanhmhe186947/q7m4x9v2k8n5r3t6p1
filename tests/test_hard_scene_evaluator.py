"""Tests for the hard-scene identity evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.evaluation.tracking.hard_scene import (
    HardSceneEvalConfig,
    build_canonical_mapping,
    compute_frame_scores,
    match_all_frames,
    run_hard_scene_evaluation,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic CVAT XML generation
# ---------------------------------------------------------------------------


def _write_cvat_xml(
    path: Path,
    tracks: dict[int, dict[int, list[float]]],
    *,
    task_name: str = "test_video",
) -> None:
    """Write a minimal CVAT-for-video 1.1 XML.

    ``tracks`` maps ``track_id -> {frame -> [x1, y1, x2, y2]}``.
    """
    parts = [
        "<annotations>"
        "<meta><task><name>" + task_name + "</name>"
        "<size>100</size></task></meta>"
    ]
    for track_id, by_frame in tracks.items():
        parts.append(f'<track id="{track_id}" label="Pig_{track_id}">')
        for frame, box in sorted(by_frame.items()):
            x1, y1, x2, y2 = box
            parts.append(
                f'<box frame="{frame}" xtl="{x1}" ytl="{y1}" '
                f'xbr="{x2}" ybr="{y2}" outside="0">'
                f'<attribute name="ID">ID_{track_id}</attribute>'
                f'<attribute name="Hidden">No</attribute>'
                f"</box>"
            )
        parts.append("</track>")
    parts.append("</annotations>")
    path.write_text("".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: canonical mapping majority vote
# ---------------------------------------------------------------------------


def test_canonical_mapping_majority_vote(tmp_path: Path) -> None:
    """Pred track IDs differ from GT IDs — mapping picks majority GT."""
    # GT: track 1 at (0,0,20,20), track 2 at (100,0,120,20) for 10 frames
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(10)},
        2: {f: [100, 0, 120, 20] for f in range(10)},
    }
    # Pred: track 7 overlaps GT 1, track 8 overlaps GT 2 (all 10 frames)
    pred_tracks = {
        7: {f: [0, 0, 20, 20] for f in range(10)},
        8: {f: [100, 0, 120, 20] for f in range(10)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    from pig_behavior.evaluation.tracking.cvat_io import parse_cvat_video_xml

    gt_by_frame = parse_cvat_video_xml(gt_xml)
    pred_by_frame = parse_cvat_video_xml(pred_xml)
    frame_results = match_all_frames(gt_by_frame, pred_by_frame, 0.30)

    gt_area_hist: dict[str, list[float]] = {}
    frame_hardness: dict[int, float] = {}
    for f in sorted(frame_results):
        h, _, _ = compute_frame_scores(frame_results[f], 0.30, gt_area_hist)
        frame_hardness[f] = h

    mapping = build_canonical_mapping(frame_results, frame_hardness, config)

    # pred track "7" should map to GT "ID_1", track "8" to GT "ID_2"
    assert mapping["7"][0] == "ID_1"
    assert mapping["8"][0] == "ID_2"


# ---------------------------------------------------------------------------
# Test 2: wrong ID flagged
# ---------------------------------------------------------------------------


def test_wrong_id_flagged(tmp_path: Path) -> None:
    """When pred track canonical != gt_id, is_id_wrong should be True."""
    # GT: 2 pigs, 5 frames
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(5)},
        2: {f: [100, 0, 120, 20] for f in range(5)},
    }
    # Pred: track 7 = GT1 on frames 0-3, then jumps to GT2 position on frame 4
    pred_tracks = {
        7: {
            0: [0, 0, 20, 20],
            1: [0, 0, 20, 20],
            2: [0, 0, 20, 20],
            3: [0, 0, 20, 20],
            4: [100, 0, 120, 20],  # wrong: canonical is GT1 but now at GT2
        },
        8: {f: [100, 0, 120, 20] for f in range(4)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    run_hard_scene_evaluation(config)

    # Read per-frame CSV and check
    df = pd.read_csv(config.output_dir / "per_frame_identity_analysis.csv")
    # Frame 4, GT ID_2 matched to pred track 7 (canonical ID_1) → wrong
    wrong_rows = df[(df["frame_idx"] == 4) & (df["is_id_wrong"] == True)]  # noqa: E712
    assert len(wrong_rows) >= 1
    assert wrong_rows.iloc[0]["gt_id"] == "ID_2"


# ---------------------------------------------------------------------------
# Test 3: two-way swap grouped into one event
# ---------------------------------------------------------------------------


def test_two_way_swap_grouped(tmp_path: Path) -> None:
    """Consecutive mutual swaps should be grouped into a single event."""
    # GT: 2 pigs, well-separated
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(20)},
        2: {f: [100, 0, 120, 20] for f in range(20)},
    }
    # Pred: track 7 normally tracks GT1, track 8 normally tracks GT2
    # Frames 0-4: correct
    # Frames 5-14: swapped (7→GT2 pos, 8→GT1 pos)
    # Frames 15-19: correct again
    pred_tracks = {
        7: {},
        8: {},
    }
    for f in range(20):
        if 5 <= f <= 14:
            # Swapped positions
            pred_tracks[7][f] = [100, 0, 120, 20]
            pred_tracks[8][f] = [0, 0, 20, 20]
        else:
            pred_tracks[7][f] = [0, 0, 20, 20]
            pred_tracks[8][f] = [100, 0, 120, 20]

    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    run_hard_scene_evaluation(config)

    events_df = pd.read_csv(config.output_dir / "swap_events.csv")
    # Should have exactly 1 swap event covering frames 5-14
    assert len(events_df) == 1
    event = events_df.iloc[0]
    assert event["start_frame"] == 5
    assert event["end_frame"] == 14
    assert event["duration_frames"] == 10
    assert event["recovered_after_event"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Test 4: long swap threshold
# ---------------------------------------------------------------------------


def test_long_swap_threshold(tmp_path: Path) -> None:
    """Events with duration >= long_swap_threshold counted as long-term."""
    n_frames = 50
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(n_frames)},
        2: {f: [100, 0, 120, 20] for f in range(n_frames)},
    }
    pred_tracks = {7: {}, 8: {}}
    # Short swap: frames 20-22 (3 frames)
    # Long swap: frames 30-44 (15 frames)
    # Correct frames: 0-19, 23-29, 45-49 = 32 frames (clear majority)
    for f in range(n_frames):
        if 20 <= f <= 22 or 30 <= f <= 44:
            pred_tracks[7][f] = [100, 0, 120, 20]
            pred_tracks[8][f] = [0, 0, 20, 20]
        else:
            pred_tracks[7][f] = [0, 0, 20, 20]
            pred_tracks[8][f] = [100, 0, 120, 20]

    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml,
        pred_xml=pred_xml,
        output_dir=tmp_path / "out",
        long_swap_threshold=10,  # events >= 10 frames are "long"
    )
    metrics = run_hard_scene_evaluation(config)

    assert metrics["total_swap_events"] == 2
    assert metrics["long_term_swap_count"] == 1  # only the 15-frame event


# ---------------------------------------------------------------------------
# Test 5: hard/critical ID accuracy
# ---------------------------------------------------------------------------


def test_hard_critical_accuracy(tmp_path: Path) -> None:
    """Per-difficulty accuracy is computed correctly from matched instances."""
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(10)},
        2: {f: [100, 0, 120, 20] for f in range(10)},
    }
    pred_tracks = {
        7: {f: [0, 0, 20, 20] for f in range(10)},
        8: {f: [100, 0, 120, 20] for f in range(10)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    metrics = run_hard_scene_evaluation(config)

    # All matched, all correct → global accuracy = 1.0
    assert metrics["global_id_accuracy"] == 1.0
    # With well-separated boxes, all frames should be easy → easy accuracy 1.0
    assert metrics["id_accuracy_easy"] == 1.0


# ---------------------------------------------------------------------------
# Test 6: missing GT is not counted as wrong ID
# ---------------------------------------------------------------------------


def test_missing_not_wrong(tmp_path: Path) -> None:
    """Missing GT instances should NOT be counted as wrong ID."""
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(10)},
        2: {f: [100, 0, 120, 20] for f in range(10)},
    }
    # Pred: only track for GT 1, no track for GT 2
    pred_tracks = {
        7: {f: [0, 0, 20, 20] for f in range(10)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    metrics = run_hard_scene_evaluation(config)

    # 10 matched (GT1), 10 missing (GT2) — no wrong IDs
    assert metrics["total_wrong_id_frames"] == 0
    assert metrics["total_missing_instances"] == 10
    # Global accuracy based on matched only → should be 1.0
    assert metrics["global_id_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test 7: compare mode creates comparison CSV
# ---------------------------------------------------------------------------


def test_incomplete_prediction_is_marked_invalid(tmp_path: Path) -> None:
    """A 10-frame prediction for a 1800-frame GT is incomplete, not a success."""
    gt_tracks = {1: {f: [0, 0, 20, 20] for f in range(1800)}}
    pred_tracks = {7: {f: [0, 0, 20, 20] for f in range(10)}}
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    metrics = run_hard_scene_evaluation(config)

    assert metrics["total_frames"] == 1800
    assert metrics["predicted_frame_count"] == 10
    assert metrics["matched_instance_count"] == 10
    assert metrics["evaluation_valid"] is False
    assert "invalid_or_incomplete_prediction" in metrics["invalid_reason"]
    assert metrics["prediction_frame_coverage"] < 0.5
    assert metrics["matched_instance_ratio"] < 0.5

    metrics_json = json.loads(
        (config.output_dir / "hard_scene_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics_json["evaluation_valid"] is False


def test_hard_frame_without_prediction_is_not_valid_success(tmp_path: Path) -> None:
    """Hard GT frames with no predictions make the evaluation incomplete."""
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(10)},
        2: {f: [0, 0, 20, 20] for f in range(10)},
    }
    pred_tracks = {
        7: {f: [0, 0, 20, 20] for f in range(4)},
        8: {f: [0, 0, 20, 20] for f in range(4)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_xml, pred_tracks)

    config = HardSceneEvalConfig(
        gt_xml=gt_xml, pred_xml=pred_xml, output_dir=tmp_path / "out"
    )
    metrics = run_hard_scene_evaluation(config)

    hard_df = pd.read_csv(config.output_dir / "hard_frame_summary.csv")
    frame_9 = hard_df[hard_df["frame_idx"] == 9].iloc[0]
    assert frame_9["difficulty"] == "hard"
    assert frame_9["num_pred"] == 0
    assert frame_9["num_missing"] == 2
    assert metrics["evaluation_valid"] is False
    assert metrics["total_missing_instances"] > 0


def test_compare_creates_comparison_csv(tmp_path: Path) -> None:
    """Compare CLI mode produces hard_scene_config_comparison.csv."""
    gt_tracks = {
        1: {f: [0, 0, 20, 20] for f in range(5)},
        2: {f: [100, 0, 120, 20] for f in range(5)},
    }
    pred_a_tracks = {
        7: {f: [0, 0, 20, 20] for f in range(5)},
        8: {f: [100, 0, 120, 20] for f in range(5)},
    }
    pred_b_tracks = {
        7: {f: [100, 0, 120, 20] for f in range(5)},  # always wrong
        8: {f: [0, 0, 20, 20] for f in range(5)},
    }

    gt_xml = tmp_path / "gt.xml"
    pred_a = tmp_path / "pred_a.xml"
    pred_b = tmp_path / "pred_b.xml"
    _write_cvat_xml(gt_xml, gt_tracks)
    _write_cvat_xml(pred_a, pred_a_tracks)
    _write_cvat_xml(pred_b, pred_b_tracks)

    from pig_behavior.evaluation.tracking.hard_scene import (
        main_compare,
    )

    output_dir = tmp_path / "compare_out"
    main_compare(
        [
            "--gt-xml",
            str(gt_xml),
            "--pred",
            f"baseline={pred_a}",
            "--pred",
            f"swapped={pred_b}",
            "--output-dir",
            str(output_dir),
        ]
    )

    comparison_csv = output_dir / "hard_scene_config_comparison.csv"
    assert comparison_csv.exists()
    df = pd.read_csv(comparison_csv)
    assert set(df["config_name"]) == {"baseline", "swapped"}
    # Baseline should have perfect accuracy
    baseline_row = df[df["config_name"] == "baseline"].iloc[0]
    assert baseline_row["global_id_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test 8: CLI Auto-mapping, custom roots, and comparison discovery
# ---------------------------------------------------------------------------


def test_main_auto_mapping_single_video(tmp_path: Path) -> None:
    """pig-tracking-hard-eval with --video only auto-resolves GT/preds."""
    gt_dir = tmp_path / "annotations" / "tracking"
    video_dir = tmp_path / "videos"
    prediction_root = tmp_path / "outputs" / "id_tracking"
    
    gt_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    prediction_root.mkdir(parents=True)
    
    # Write files
    (video_dir / "pig_video_1.mp4").write_text("")
    gt_tracks = {1: {f: [0, 0, 20, 20] for f in range(5)}}
    _write_cvat_xml(gt_dir / "pig_video_1.xml", gt_tracks)
    
    pred_dir = prediction_root / "pig_video_1"
    pred_dir.mkdir(parents=True)
    _write_cvat_xml(pred_dir / "pig_video_1_annotations_cvat_video_1_1.xml", gt_tracks)
    
    from pig_behavior.evaluation.tracking.hard_scene import main
    output_dir = tmp_path / "out"
    main([
        "--video", "pig_video_1",
        "--gt-dir", str(gt_dir),
        "--video-dir", str(video_dir),
        "--prediction-root", str(prediction_root),
        "--output-dir", str(output_dir),
    ])
    
    assert (output_dir / "per_frame_identity_analysis.csv").exists()
    assert (output_dir / "hard_scene_metrics.json").exists()


def test_main_missing_xml_errors(tmp_path: Path) -> None:
    """pig-tracking-hard-eval raises FileNotFoundError if prediction XML is missing."""
    gt_dir = tmp_path / "annotations" / "tracking"
    video_dir = tmp_path / "videos"
    prediction_root = tmp_path / "outputs" / "id_tracking"
    
    gt_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    prediction_root.mkdir(parents=True)
    
    (video_dir / "pig_video_1.mp4").write_text("")
    gt_tracks = {1: {f: [0, 0, 20, 20] for f in range(5)}}
    _write_cvat_xml(gt_dir / "pig_video_1.xml", gt_tracks)
    
    from pig_behavior.evaluation.tracking.hard_scene import main
    output_dir = tmp_path / "out"
    
    with pytest.raises(FileNotFoundError):
        main([
            "--video", "pig_video_1",
            "--gt-dir", str(gt_dir),
            "--video-dir", str(video_dir),
            "--prediction-root", str(prediction_root),
            "--output-dir", str(output_dir),
        ])


def test_main_multiple_videos(tmp_path: Path) -> None:
    """pig-tracking-hard-eval with no video path evaluates all matching pairs."""
    gt_dir = tmp_path / "annotations" / "tracking"
    video_dir = tmp_path / "videos"
    prediction_root = tmp_path / "outputs" / "id_tracking"
    
    gt_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    prediction_root.mkdir(parents=True)
    
    gt_tracks = {1: {f: [0, 0, 20, 20] for f in range(5)}}
    
    for i in (1, 2):
        stem = f"pig_video_{i}"
        (video_dir / f"{stem}.mp4").write_text("")
        _write_cvat_xml(gt_dir / f"{stem}.xml", gt_tracks)
        
        pred_dir = prediction_root / stem
        pred_dir.mkdir(parents=True)
        _write_cvat_xml(pred_dir / f"{stem}_annotations_cvat_video_1_1.xml", gt_tracks)
        
    from pig_behavior.evaluation.tracking.hard_scene import main
    output_dir = tmp_path / "out"
    main([
        "--gt-dir", str(gt_dir),
        "--video-dir", str(video_dir),
        "--prediction-root", str(prediction_root),
        "--output-dir", str(output_dir),
    ])
    
    assert (output_dir / "pig_video_1" / "per_frame_identity_analysis.csv").exists()
    assert (output_dir / "pig_video_2" / "per_frame_identity_analysis.csv").exists()


def test_main_compare_auto_mapping(tmp_path: Path) -> None:
    """pig-tracking-hard-eval-compare auto-maps the latest benchmark predictions."""
    gt_dir = tmp_path / "annotations" / "tracking"
    video_dir = tmp_path / "videos"
    prediction_root = tmp_path / "outputs" / "id_tracking"
    
    gt_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    prediction_root.mkdir(parents=True)
    
    # Setup video & GT
    (video_dir / "pig_video_1.mp4").write_text("")
    gt_tracks = {1: {f: [0, 0, 20, 20] for f in range(5)}}
    _write_cvat_xml(gt_dir / "pig_video_1.xml", gt_tracks)
    
    # Setup runs: run_1 (older), run_2 (latest)
    run_1 = prediction_root / "tracking_detector_benchmark" / "20260622_120000"
    run_2 = prediction_root / "tracking_detector_benchmark" / "20260622_130000"
    
    # older run_1 has nothing
    run_1.mkdir(parents=True)
    
    # latest run_2 has yolov8 and yolov26 configs
    for detector in ("yolov8", "yolov26"):
        pred_dir = run_2 / detector / "iou0_area1_condarea1_merge0" / "pig_video_1"
        pred_dir.mkdir(parents=True)
        xml_name = "pig_video_1_annotations_cvat_video_1_1.xml"
        _write_cvat_xml(pred_dir / xml_name, gt_tracks)
        
    from pig_behavior.evaluation.tracking.hard_scene import main_compare
    output_dir = tmp_path / "compare_out"
    
    # Test Auto-discover of latest run (run_2)
    main_compare([
        "--video", "pig_video_1",
        "--gt-dir", str(gt_dir),
        "--video-dir", str(video_dir),
        "--prediction-root", str(prediction_root),
        "--output-dir", str(output_dir),
    ])
    
    # Config directories: "yolov8_iou0_area1_condarea1_merge0",
    # "yolov26_iou0_area1_condarea1_merge0"
    assert (output_dir / "yolov8_iou0_area1_condarea1_merge0").exists()
    assert (output_dir / "yolov26_iou0_area1_condarea1_merge0").exists()
    assert (output_dir / "hard_scene_config_comparison.csv").exists()
    
    # Test explicit --run-dir pointing to run_1
    # (which is empty and should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        main_compare([
            "--video", "pig_video_1",
            "--gt-dir", str(gt_dir),
            "--video-dir", str(video_dir),
            "--prediction-root", str(prediction_root),
            "--run-dir", str(run_1),
            "--output-dir", str(tmp_path / "compare_out_explicit"),
        ])
