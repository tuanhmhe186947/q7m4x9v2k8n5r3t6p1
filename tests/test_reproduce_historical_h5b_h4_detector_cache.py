from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tracking"
    / "reproduce_historical_h5b_h4_detector_cache.py"
)
SPEC = importlib.util.spec_from_file_location("historical_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_detector_configuration() -> None:
    assert MODULE.CONFIDENCE == 0.20
    assert MODULE.MAX_RAW_DETECTIONS == 64
    assert MODULE.NMS_IOU == 0.50
    assert MODULE.IMAGE_SIZE == 640
    assert MODULE.EXPECTED_TOTAL == 23400
    assert MODULE.PRODUCER_TOPOLOGY == (
        "CURRENT_PRODUCER_SEMANTICALLY_EQUIVALENT"
    )


def test_repeat_policy_is_predeclared_and_bounded() -> None:
    assert MODULE.FIXED_REPEAT_FRAMES[0] == 0
    assert MODULE.FIXED_REPEAT_FRAMES[-1] == 1799
    assert len(MODULE.FIXED_REPEAT_FRAMES) == len(
        set(MODULE.FIXED_REPEAT_FRAMES)
    )
    assert all(0 <= frame < 1800 for frame in MODULE.FIXED_REPEAT_FRAMES)


def test_pairwise_iou_contract() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    values = MODULE._pairwise_iou(boxes)  # noqa: SLF001
    assert np.isclose(values.max(), 1.0 / 3.0)


def test_no_tracker_repair_evaluator_prediction_or_mp4_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".track(" not in source
    assert "evaluate_tracking" not in source
    assert "apply_offline_repair" not in source
    assert "VideoWriter" not in source
    assert "annotations_cvat" not in source
