from __future__ import annotations

import os

import pytest

# Skip the entire module if dependencies are missing in CI
try:
    import backoff  # noqa: F401
    import cv2
    import inference_sdk  # noqa: F401
    import numpy as np

    from pig_behavior.roboflow_client import detect_pigs_roboflow
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="Missing heavy dependencies (cv2, backoff, or inference-sdk) in CI environment."
    )


@pytest.mark.skipif(
    not os.environ.get("ROBOFLOW_API_KEY"),
    reason="ROBOFLOW_API_KEY environment variable not set."
)
def test_roboflow_workflow_smoke() -> None:
    """Smoke test to verify that the Roboflow Workflow client connects, runs, and returns correct keys."""
    # Load background image as a sample input
    sample_path = os.path.join("data", "annotations", "scene", "background.png")
    assert os.path.exists(sample_path), f"Sample image not found at {sample_path}"

    frame = cv2.imread(sample_path)
    assert frame is not None, f"Failed to load sample image from {sample_path}"
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected BGR frame, got shape {frame.shape}"

    # Run the workflow
    result = detect_pigs_roboflow(frame)

    # Assert expected output keys exist and types are correct
    assert "count_objects" in result, "Response missing 'count_objects' key"
    assert "predictions" in result, "Response missing 'predictions' key"
    assert "visualized_frame" in result, "Response missing 'visualized_frame' key"

    assert isinstance(result["count_objects"], (int, float)), "count_objects should be a number"
    assert isinstance(result["predictions"], list), "predictions should be a list"
    assert isinstance(result["visualized_frame"], np.ndarray), "visualized_frame should be a numpy array"

    assert result["visualized_frame"].ndim == 3, "visualized_frame should be a 3D image array"
    assert result["visualized_frame"].shape[2] == 3, "visualized_frame should have 3 color channels (BGR)"

    # Print success diagnostic info
    print("\n[*] Smoke test succeeded!")
    print(f"[*] Detected pig count: {result['count_objects']}")
    print(f"[*] Bounding boxes found: {len(result['predictions'])}")
    if result["predictions"]:
        print(f"[*] Sample box class: {result['predictions'][0].get('class')}")
        print(f"[*] Sample box confidence: {result['predictions'][0].get('score')}")
