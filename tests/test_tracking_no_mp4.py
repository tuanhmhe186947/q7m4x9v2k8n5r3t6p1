from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pig_behavior.tracking as tracking
from pig_behavior.evaluation.tracking import hard_scene
from pig_behavior.evaluation.tracking.artifact_guard import (
    assert_no_mp4_artifacts,
)
from pig_behavior.evaluation.tracking.config import (
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.hard_scene import HardSceneEvalConfig
from pig_behavior.evaluation.tracking.pipeline import run_pipeline
from pig_behavior.tracking import runner
from pig_behavior.tracking.cli import parse_args
from pig_behavior.tracking.config import TrackingConfig


def _write_single_track_xml(path: Path) -> None:
    path.write_text(
        (
            "<annotations>"
            "<meta><task><name>pig</name><size>1</size></task></meta>"
            '<track id="0" label="Pig_1">'
            '<box frame="0" xtl="0" ytl="0" xbr="20" ybr="20" '
            'outside="0">'
            '<attribute name="ID">ID_1</attribute>'
            '<attribute name="Hidden">No</attribute>'
            "</box></track></annotations>"
        ),
        encoding="utf-8",
    )


def test_tracking_cli_exposes_no_output_video_flag() -> None:
    assert parse_args([]).no_output_video is False
    assert parse_args(["--no-output-video"]).no_output_video is True


def test_renderer_is_not_called_when_output_video_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_renderer(*args, **kwargs) -> int:
        raise AssertionError("renderer must not run")

    monkeypatch.setattr(runner, "render_annotation_video", fail_renderer)
    output_video = tmp_path / "tracked.mp4"
    cfg = TrackingConfig(write_output_video=False)

    runner._render_output_video(cfg, output_video, [], frames_written=1)

    assert not output_video.exists()


def test_evaluation_emits_xml_and_metrics_without_mp4(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"")
    gt_xml = tmp_path / "input.xml"
    _write_single_track_xml(gt_xml)
    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"weights")
    prediction_root = tmp_path / "predictions"
    report_root = tmp_path / "reports"

    def fake_run_tracking(cfg: TrackingConfig) -> SimpleNamespace:
        assert cfg.write_output_video is False
        pred_xml = cfg.output_dir / "prediction.xml"
        pred_xml.parent.mkdir(parents=True, exist_ok=True)
        _write_single_track_xml(pred_xml)
        return SimpleNamespace(
            cvat_video_xml=pred_xml,
            output_video=cfg.output_dir / "disabled.mp4",
        )

    monkeypatch.setattr(tracking, "run_tracking", fake_run_tracking)
    config = TrackingEvaluationPipelineConfig(
        video_path=input_video,
        gt_xml=gt_xml,
        prediction_root=prediction_root,
        output_root=report_root,
        weights_path=weights_path,
        force_track=True,
        expected_video_count=1,
    )

    _, metrics_df, run_dir = run_pipeline(config)

    assert (prediction_root / "prediction.xml").exists()
    assert not metrics_df.empty
    assert (run_dir / "tracking_metrics.csv").exists()
    assert (run_dir / "tracking_runtime_telemetry.csv").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "artifact_manifest.json").exists()
    assert not list(prediction_root.rglob("*.mp4"))
    assert not list(report_root.rglob("*.mp4"))


def test_recursive_mp4_guard_rejects_stale_artifact(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "stale.MP4").write_bytes(b"")

    try:
        assert_no_mp4_artifacts(tmp_path, context="test experiment")
    except RuntimeError as exc:
        assert "nested" in str(exc)
        assert "stale.MP4" in str(exc)
    else:
        raise AssertionError("MP4 artifact guard must fail closed")


def test_hard_scene_video_rendering_is_opt_in(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    input_video = tmp_path / "input.mp4"
    output_dir = tmp_path / "hard_scene"
    _write_single_track_xml(gt_xml)
    _write_single_track_xml(pred_xml)
    input_video.write_bytes(b"")

    def fail_renderer(*args, **kwargs) -> None:
        raise AssertionError("hard-scene renderer must be opt-in")

    monkeypatch.setattr(hard_scene, "render_overlay_video", fail_renderer)
    config = HardSceneEvalConfig(
        gt_xml=gt_xml,
        pred_xml=pred_xml,
        video_path=input_video,
        output_dir=output_dir,
    )

    hard_scene.run_hard_scene_evaluation(config)

    assert (output_dir / "hard_scene_metrics.json").exists()
    assert (output_dir / "per_frame_identity_analysis.csv").exists()
    assert not list(output_dir.rglob("*.mp4"))
