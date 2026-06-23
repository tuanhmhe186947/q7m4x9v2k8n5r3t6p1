#!/usr/bin/env python3
"""Roboflow Workflows client integration for pig detection."""

from __future__ import annotations

import base64
import os
import sys
from typing import Any

import backoff
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPClientError
from numpy.typing import NDArray


class RoboflowError(Exception):
    """Custom exception for Roboflow API errors."""
    pass


def get_roboflow_client(api_key: str | None = None) -> InferenceHTTPClient:
    """Initialize the InferenceHTTPClient using environment variables or passed key."""
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RoboflowError(
            "Roboflow API key not found. Please set the ROBOFLOW_API_KEY environment variable "
            "or pass it explicitly."
        )
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=key
    )


@backoff.on_exception(
    backoff.expo,
    (HTTPClientError, Exception),
    max_tries=3,
    factor=2,
    logger=None
)
def run_roboflow_workflow_with_retry(
    client: InferenceHTTPClient,
    workspace_name: str,
    workflow_id: str,
    image_b64: str,
) -> dict[str, Any]:
    """Execute Roboflow workflow with exponential backoff on retryable HTTP errors."""
    # Note: inference-sdk uses requests internally which handles connections,
    # and we wrap it with backoff to handle transient network issues or rate limits.
    try:
        results = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": image_b64}
        )
        if not results or len(results) == 0:
            raise RoboflowError("Received empty response list from Roboflow Workflow execution.")
        
        # Grounded response structure: it is a list of results, we return the first one
        return results[0]
    except Exception as e:
        if isinstance(e, RoboflowError):
            raise
        raise RoboflowError(f"Roboflow Workflow execution failed: {e}") from e


def detect_pigs_roboflow(
    frame: NDArray[np.uint8],
    api_key: str | None = None,
    workspace_name: str = "projectdetectpigbehaviorvideoprocess",
    workflow_id: str = "detect-count-and-visualize-3",
) -> dict[str, Any]:
    """Encode the image frame and execute Roboflow workflow.

    Args:
        frame: OpenCV image frame (NDArray of shape [H, W, 3]).
        api_key: Optional Roboflow API key.
        workspace_name: Roboflow workspace slug.
        workflow_id: Roboflow workflow slug.

    Returns:
        A dict containing parsed results:
            - 'count_objects': int (number of detected pigs)
            - 'predictions': list of dicts (parsed bbox predictions)
            - 'visualized_frame': NDArray (OpenCV decoded BGR image)
    """
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected 3D BGR image, got shape {frame.shape}"
    
    # 1. Base64 encode the frame to JPEG buffer
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RoboflowError("Failed to encode frame to JPEG for Roboflow API.")
    
    img_b64 = base64.b64encode(buf).decode("utf-8")
    
    # 2. Get client and run workflow with retry
    client = get_roboflow_client(api_key)
    res = run_roboflow_workflow_with_retry(
        client=client,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        image_b64=img_b64
    )
    
    # 3. Parse defensively based on real response types from describe_interface/run tests
    count_objects = res.get("count_objects", 0)
    
    # Standardize predictions list to box formats
    raw_predictions_container = res.get("predictions", {})
    raw_predictions = []
    if isinstance(raw_predictions_container, dict):
        raw_predictions = raw_predictions_container.get("predictions", [])
    elif isinstance(raw_predictions_container, list):
        raw_predictions = raw_predictions_container
        
    parsed_predictions = []
    for pred in raw_predictions:
        # Convert center (x, y) coordinates to top-left / bottom-right [x1, y1, x2, y2]
        x = pred.get("x", 0.0)
        y = pred.get("y", 0.0)
        w = pred.get("width", 0.0)
        h = pred.get("height", 0.0)
        
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        
        parsed_predictions.append({
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "score": float(pred.get("confidence", 0.0)),
            "class": pred.get("class", "Pig"),
            "class_id": int(pred.get("class_id", 0))
        })
        
    # Sort predictions by score descending
    parsed_predictions.sort(key=lambda x: x["score"], reverse=True)
    
    # Decode output image
    visualized_frame = None
    output_image_data = res.get("output_image")
    if output_image_data:
        # If output_image is returned directly as a string or within a dict
        b64_str = ""
        if isinstance(output_image_data, str):
            b64_str = output_image_data
        elif isinstance(output_image_data, dict):
            b64_str = output_image_data.get("value", "")
            
        if b64_str:
            try:
                img_bytes = base64.b64decode(b64_str)
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                visualized_frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[!] Warning: Failed to decode output_image base64: {e}", file=sys.stderr)
                
    if visualized_frame is None:
        # Fallback to copy of original frame if visualization decoding failed
        visualized_frame = frame.copy()
        
    return {
        "count_objects": count_objects,
        "predictions": parsed_predictions,
        "visualized_frame": visualized_frame
    }
