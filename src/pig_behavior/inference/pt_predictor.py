"""Inference utilities for the PyTorch behavior sequence classifier."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pig_behavior.models.behavior_sequence import (
    BehaviorFrameSample,
    BehaviorSequenceClassifier,
)


@dataclass(slots=True)
class PTDetection:
    """Kept for API compatibility; behavior classification has no detections."""

    label: str
    confidence: float
    bbox_xyxy: list[float]
    class_id: int


@dataclass(slots=True)
class PTPrediction:
    """Prediction result returned by the behavior sequence model."""

    task: str
    latency_ms: float
    predicted_label: str | None
    confidence: float | None
    scores: dict[str, float]
    detections: list[PTDetection]


class PTModelService:
    """Lazy wrapper for the custom PyTorch behavior classifier."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._classifier = BehaviorSequenceClassifier(model_path)

    @property
    def model_loaded(self) -> bool:
        """Return whether the classifier has been loaded."""
        return self._classifier.loaded

    @property
    def model_available(self) -> bool:
        """Return whether the classifier path exists on disk."""
        return self.model_path.exists()

    def load(self) -> None:
        """Load the classifier if needed."""
        self._classifier.load()

    def predict_path(
        self,
        image_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
    ) -> PTPrediction:
        """Run a padded 6-frame prediction from a single image path."""
        del confidence_threshold

        from PIL import Image

        with Image.open(image_path) as image:
            crop_rgb = np.asarray(image.convert("RGB"))
        return self._predict_single_crop(crop_rgb, bbox=None)

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        confidence_threshold: float = 0.25,
    ) -> PTPrediction:
        """Run a padded 6-frame prediction from uploaded image bytes."""
        del confidence_threshold

        from pig_behavior.inference.tflite_predictor import (
            image_from_bytes,
            validate_bbox,
        )

        with image_from_bytes(image_bytes) as image:
            image = image.convert("RGB")
            bbox_values = None
            if bbox is not None:
                bbox_values = validate_bbox(bbox)
                image = image.crop(tuple(int(round(value)) for value in bbox_values))
            crop_rgb = np.asarray(image)
        return self._predict_single_crop(crop_rgb, bbox=bbox)

    def _predict_single_crop(
        self,
        crop_rgb: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
    ) -> PTPrediction:
        start = time.perf_counter()
        height, width = crop_rgb.shape[:2]
        if bbox is None:
            features = [0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            bbox_xyxy = [0.0, 0.0, float(width), float(height)]
        else:
            x1, y1, x2, y2 = bbox
            frame_width = max(float(x2), float(width), 1.0)
            frame_height = max(float(y2), float(height), 1.0)
            features = [
                ((x1 + x2) / 2.0) / frame_width,
                ((y1 + y2) / 2.0) / frame_height,
                (x2 - x1) / frame_width,
                (y2 - y1) / frame_height,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]

        sample = BehaviorFrameSample(
            crop_rgb=crop_rgb,
            features=features,
            frame_index=0,
            bbox_xyxy=bbox_xyxy,
        )
        prediction = self._classifier.predict([sample])
        latency_ms = (time.perf_counter() - start) * 1000
        return PTPrediction(
            task="classify_sequence_padded",
            latency_ms=latency_ms,
            predicted_label=prediction.label,
            confidence=prediction.confidence,
            scores=prediction.scores,
            detections=[],
        )


def run_pt_inference(
    image_path: str | Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.25,
) -> PTPrediction:
    """Run one CLI prediction using the behavior sequence model."""
    service = PTModelService(model_path)
    prediction = service.predict_path(
        image_path,
        confidence_threshold=confidence_threshold,
    )
    print_pt_prediction(image_path, model_path, prediction)
    return prediction


def print_pt_prediction(
    image_path: str | Path,
    model_path: Path,
    prediction: PTPrediction,
) -> None:
    """Print a behavior prediction in a readable CLI format."""
    print()
    print("-" * 72)
    print(f"Image:   {image_path}")
    print(f"Model:   {model_path}")
    print(f"Task:    {prediction.task}")
    print(f"Latency: {prediction.latency_ms:.2f} ms")
    print("-" * 72)
    print(
        "Prediction: "
        f"{prediction.predicted_label} ({prediction.confidence or 0.0:.1%})"
    )
    print()
    print(f"{'Class':<24} {'Confidence':>10}")
    print("-" * 37)
    for label, confidence in sorted(
        prediction.scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{label:<24} {confidence:>9.1%}")
    print()
