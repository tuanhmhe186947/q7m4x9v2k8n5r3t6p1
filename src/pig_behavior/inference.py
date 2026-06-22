"""Backward-compatible imports for TFLite inference services."""

from __future__ import annotations

from pig_behavior.inference.tflite_predictor import (
    PredictionScores,
    image_from_bytes,
    load_interpreter,
    predict,
    preprocess_image,
    preprocess_pil_image,
    resolve_tflite_path,
    run_inference,
    validate_bbox,
)

__all__ = [
    "PredictionScores",
    "image_from_bytes",
    "load_interpreter",
    "predict",
    "preprocess_image",
    "preprocess_pil_image",
    "resolve_tflite_path",
    "run_inference",
    "validate_bbox",
]
