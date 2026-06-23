import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.evaluation.tracking.benchmarking import (  # noqa: E402
    iter_detector_benchmark_configs,
)
from pig_behavior.evaluation.tracking.config import (  # noqa: E402
    TrackingEvaluationPipelineConfig,
)
from pig_behavior.evaluation.tracking.pipeline import (  # noqa: E402
    tracking_rule_overrides,
)


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
    )

    overrides = tracking_rule_overrides(config)

    assert overrides["device"] == "0"
    assert overrides["half"] is True
    assert overrides["USE_IOU_FALLBACK"] is True
    assert overrides["USE_AREA_OCCLUSION_FREEZE"] is True
    assert overrides["USE_CONDITIONAL_AREA_OCCLUSION_FREEZE"] is False
    assert overrides["USE_MERGED_BOX_SPLIT"] is True
