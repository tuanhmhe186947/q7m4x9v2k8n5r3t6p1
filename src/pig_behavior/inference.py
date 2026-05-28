"""Inference utilities for the pig behavior classifier."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from pig_behavior.config import EXPORT_DIR, TABULAR_FEATURES, TrainConfig

PredictionScores = dict[str, float]


def load_interpreter(tflite_path: Path | None = None) -> tf.lite.Interpreter:
    """Load and allocate a TFLite interpreter."""
    if tflite_path is None:
        int8_path = EXPORT_DIR / "model_int8.tflite"
        fp32_path = EXPORT_DIR / "model_fp32.tflite"
        tflite_path = int8_path if int8_path.exists() else fp32_path

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"TFLite model not found at {tflite_path}. Run export first."
        )

    print(f"[infer] Loading TFLite model from {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(
    image_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load, crop, resize, and normalize one image."""
    image = Image.open(image_path).convert("RGB")

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        image = image.crop((int(x1), int(y1), int(x2), int(y2)))

    image = image.resize((target_size[1], target_size[0]))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(
    interpreter: tf.lite.Interpreter,
    image: np.ndarray,
    tabular: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> tuple[PredictionScores, float]:
    """Run one prediction and return class scores plus latency."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        name = detail["name"].lower()
        if "tabular" in name:
            if tabular is None:
                raise ValueError(
                    "The model expects tabular input, but none was provided."
                )
            value = tabular
        else:
            value = image
        interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))

    start = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    probabilities = (
        output if np.isclose(output.sum(), 1.0, atol=1e-3) else _softmax(output)
    )

    labels = labels or [f"class_{idx}" for idx in range(len(probabilities))]
    scores = {
        label: float(probability)
        for label, probability in zip(labels, probabilities, strict=False)
    }
    return scores, latency_ms


def run_inference(
    cfg: TrainConfig,
    image_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    tabular_features: list[float] | None = None,
    tflite_path: Path | None = None,
) -> None:
    """Run the full single-image inference workflow and print the result."""
    interpreter = load_interpreter(tflite_path)
    image = preprocess_image(image_path, bbox, cfg.image_size)

    tabular = None
    if tabular_features is not None:
        if len(tabular_features) != len(TABULAR_FEATURES):
            raise ValueError(
                f"Expected {len(TABULAR_FEATURES)} tabular values, "
                f"got {len(tabular_features)}."
            )
        tabular = np.asarray([tabular_features], dtype=np.float32)

    scores, latency_ms = predict(interpreter, image, tabular, cfg.labels)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted_label, predicted_confidence = sorted_scores[0]

    print()
    print("-" * 56)
    print(f"Image:   {image_path}")
    if bbox:
        print(f"BBox:    ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
    print(f"Latency: {latency_ms:.2f} ms")
    print("-" * 56)
    print(f"Prediction: {predicted_label} ({predicted_confidence:.1%})")
    print()
    print(f"{'Class':<16} {'Confidence':>10}")
    print("-" * 29)
    for label, confidence in sorted_scores:
        bar = "#" * int(confidence * 20)
        print(f"{label:<16} {confidence:>9.1%} {bar}")
    print()


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
