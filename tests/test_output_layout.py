from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.evaluation.tracking.assets import find_prediction_xml  # noqa: E402
from pig_behavior.tracking.config import (  # noqa: E402
    TrackingConfig,
    resolve_output_paths,
)


def test_find_prediction_xml_supports_mode_first_layout(tmp_path: Path) -> None:
    video_stem = "Pigs291119_000263_30fps"
    pred_xml = tmp_path / "hybrid_bytetrack" / video_stem / "annotations_cvat_video_1_1.xml"
    pred_xml.parent.mkdir(parents=True)
    pred_xml.write_text("<annotations />", encoding="utf-8")

    assert (
        find_prediction_xml(
            video_stem,
            tmp_path,
            preferred_mode="hybrid_bytetrack",
        )
        == pred_xml
    )


def test_find_prediction_xml_supports_mode_scoped_root(tmp_path: Path) -> None:
    video_stem = "Pigs291119_000263_30fps"
    pred_root = tmp_path / "hybrid_bytetrack" / "20260626_120000"
    pred_xml = pred_root / video_stem / "annotations_cvat_video_1_1.xml"
    pred_xml.parent.mkdir(parents=True)
    pred_xml.write_text("<annotations />", encoding="utf-8")

    assert (
        find_prediction_xml(
            video_stem,
            pred_root,
            preferred_mode="hybrid_bytetrack",
        )
        == pred_xml
    )


def test_resolve_output_paths_avoids_duplicate_mode_segment(tmp_path: Path) -> None:
    cfg = TrackingConfig(
        video_path=tmp_path / "Pigs291119_000263_30fps.mp4",
        output_dir=tmp_path / "pred" / "hybrid_bytetrack" / "20260626_120000",
        mode="hybrid_bytetrack",
    )

    output_video, *_rest = resolve_output_paths(cfg)

    assert output_video.parent == cfg.output_dir / "Pigs291119_000263_30fps"


def test_resolve_output_paths_groups_direct_runs_by_mode(tmp_path: Path) -> None:
    cfg = TrackingConfig(
        video_path=tmp_path / "Pigs291119_000263_30fps.mp4",
        output_dir=tmp_path / "pred",
        mode="hybrid_bytetrack",
    )

    output_video, *_rest = resolve_output_paths(cfg)

    assert output_video.parent == (
        tmp_path / "pred" / "hybrid_bytetrack" / "Pigs291119_000263_30fps"
    )
