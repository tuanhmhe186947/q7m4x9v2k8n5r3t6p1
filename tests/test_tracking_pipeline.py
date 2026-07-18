import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.evaluation.tracking.assets import find_prediction_xml  # noqa: E402
from pig_behavior.evaluation.tracking.benchmarking import (  # noqa: E402
    iter_detector_benchmark_configs,
    rank_aggregate_benchmark_rows,
)
from pig_behavior.evaluation.tracking.config import (  # noqa: E402
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.pipeline import (  # noqa: E402
    tracking_rule_overrides,
)
from pig_behavior.output_layout import prediction_xml_candidates  # noqa: E402


def test_find_prediction_xml_prefers_mode_scoped_tracker_output(
    tmp_path: Path,
) -> None:
    video_stem = "Pigs291119_000263_30fps"
    pred_xml = (
        tmp_path
        / video_stem
        / "hybrid_bytetrack"
        / "annotations_cvat_video_1_1.xml"
    )
    pred_xml.parent.mkdir(parents=True)
    pred_xml.write_text("<annotations />", encoding="utf-8")

    assert find_prediction_xml(video_stem, tmp_path) == pred_xml


def test_find_prediction_xml_prefers_requested_mode_when_outputs_coexist(
    tmp_path: Path,
) -> None:
    video_stem = "Pigs291119_000263_30fps"
    raw_xml = (
        tmp_path
        / video_stem
        / "bytetrack_raw"
        / "annotations_cvat_video_1_1.xml"
    )
    hybrid_xml = (
        tmp_path
        / video_stem
        / "hybrid_bytetrack"
        / "annotations_cvat_video_1_1.xml"
    )
    raw_xml.parent.mkdir(parents=True)
    hybrid_xml.parent.mkdir(parents=True)
    raw_xml.write_text("<annotations />", encoding="utf-8")
    hybrid_xml.write_text("<annotations />", encoding="utf-8")

    assert (
        find_prediction_xml(
            video_stem,
            tmp_path,
            preferred_mode="hybrid_bytetrack",
        )
        == hybrid_xml
    )


def test_tracking_eval_defaults_match_hybrid_baseline() -> None:
    cfg = TrackingEvaluationPipelineConfig()

    assert cfg.tracking_mode == "hybrid_bytetrack"
    assert cfg.USE_IOU_FALLBACK is False
    assert cfg.USE_AREA_OCCLUSION_FREEZE is False
    assert cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE is False
    assert cfg.USE_MERGED_BOX_SPLIT is False


def test_find_prediction_xml_does_not_map_removed_bytetrack_alias(
    tmp_path: Path,
) -> None:
    video_stem = "Pigs291119_000263_30fps"
    candidates = prediction_xml_candidates(
        tmp_path,
        video_stem,
        preferred_mode="bytetrack",
    )

    assert (
        tmp_path
        / "hybrid_bytetrack"
        / video_stem
        / "annotations_cvat_video_1_1.xml"
    ) not in candidates


def test_detector_benchmark_configs_isolate_v8_and_v26_outputs(tmp_path: Path) -> None:
    v8_weights = tmp_path / "pig_detector_yolov8.pt"
    v26_weights = tmp_path / "pig_detector_yolov26.pt"
    v8_weights.write_bytes(b"v8")
    v26_weights.write_bytes(b"v26")
    config = TrackingEvaluationPipelineConfig(
        weights_path=v8_weights,
        weights_v26_path=v26_weights,
        prediction_root=tmp_path / "predictions",
        output_root=tmp_path / "reports",
        device="0",
        half=True,
        force_track=True,
        run_missing_tracker=True,
    )

    configs = iter_detector_benchmark_configs(
        config,
        tmp_path / "benchmark_reports",
        tmp_path / "benchmark_predictions",
    )

    assert [cfg.detector_name for cfg in configs] == ["yolov8", "yolov26"]
    assert [cfg.weights_path for cfg in configs] == [v8_weights, v26_weights]
    assert configs[0].prediction_root.name == "yolov8"
    assert configs[1].prediction_root.name == "yolov26"
    assert configs[0].output_root.name == "yolov8"
    assert configs[1].output_root.name == "yolov26"
    assert all(cfg.force_track for cfg in configs)
    assert all(cfg.run_missing_tracker for cfg in configs)


def test_tracking_rule_overrides_forwards_gpu_options() -> None:
    config = TrackingEvaluationPipelineConfig(
        device="0",
        half=True,
        USE_IOU_FALLBACK=True,
        USE_AREA_OCCLUSION_FREEZE=True,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=False,
        USE_MERGED_BOX_SPLIT=True,
        profile_overrides={
            "enable_offline_smoothing": False,
            "smooth_boxes": True,
            "refine_boxes": False,
        },
    )

    overrides = tracking_rule_overrides(config)

    assert overrides["device"] == "0"
    assert overrides["half"] is True
    assert overrides["USE_IOU_FALLBACK"] is True
    assert overrides["USE_AREA_OCCLUSION_FREEZE"] is True
    assert overrides["USE_CONDITIONAL_AREA_OCCLUSION_FREEZE"] is False
    assert overrides["USE_MERGED_BOX_SPLIT"] is True
    assert overrides["enable_offline_smoothing"] is False
    assert overrides["smooth_boxes"] is True
    assert overrides["refine_boxes"] is False
    assert overrides["overrides"] == set(overrides) - {"overrides"}


def test_rank_aggregate_benchmark_rows_prioritizes_hota_then_identity() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "row_type": "PER_VIDEO",
                "combo": "ignored",
                "remapped_hota_pct": 100.0,
            },
            {
                "row_type": "ALL",
                "combo": "higher_mota",
                "remapped_hota_pct": 94.0,
                "remapped_idf1_pct": 99.0,
                "remapped_mota_pct": 99.0,
                "remapped_idsw": 0,
            },
            {
                "row_type": "ALL",
                "combo": "best_hota",
                "remapped_hota_pct": 95.0,
                "remapped_idf1_pct": 96.0,
                "remapped_mota_pct": 97.0,
                "remapped_idsw": 1,
            },
        ]
    )

    ranked = rank_aggregate_benchmark_rows(summary_df)

    assert ranked["combo"].tolist() == ["best_hota", "higher_mota"]
