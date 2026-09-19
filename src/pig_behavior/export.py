"""Model export helpers for edge deployment."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import tensorflow as tf

from pig_behavior.config import (
    CHECKPOINT_DIR,
    EXPORT_DIR,
    TABULAR_FEATURES,
    TrainConfig,
)


def _ensure_dir() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _model_size_mb(path: Path) -> float:
    """Return a file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def export_tflite(
    cfg: TrainConfig,
    model_path: Path | None = None,
    representative_ds: tf.data.Dataset | None = None,
) -> Path:
    """Convert a trained Keras model to TFLite."""
    _ensure_dir()
    model_path = model_path or CHECKPOINT_DIR / "final_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first or pass model_path."
        )

    print(f"[export] Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()

    fp32_path = EXPORT_DIR / "model_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)
    fp32_size = _model_size_mb(fp32_path)
    print(f"[export] FP32 TFLite saved: {fp32_path} ({fp32_size:.2f} MB)")

    if not cfg.quantize:
        return fp32_path

    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_ds is not None:
        converter_q.representative_dataset = _representative_dataset(representative_ds)

    tflite_int8 = converter_q.convert()
    int8_path = EXPORT_DIR / "model_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    int8_size = _model_size_mb(int8_path)
    print(f"[export] INT8 TFLite saved: {int8_path} ({int8_size:.2f} MB)")
    print(
        "[export] Size reduction: "
        f"{fp32_size:.2f} MB to {int8_size:.2f} MB "
        f"({fp32_size / max(int8_size, 0.01):.1f}x)"
    )
    return int8_path


def _representative_dataset(
    dataset: tf.data.Dataset,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Build a TFLite representative dataset generator."""

    def generator() -> Iterator[list[np.ndarray]]:
        for inputs, _labels in dataset.take(100):
            if isinstance(inputs, dict):
                yield [
                    inputs["image_input"].numpy().astype(np.float32),
                    inputs["tabular_input"].numpy().astype(np.float32),
                ]
            else:
                yield [inputs.numpy().astype(np.float32)]

    return generator


def export_onnx(
    cfg: TrainConfig,
    model_path: Path | None = None,
) -> Path | None:
    """Convert a trained Keras model to ONNX when enabled."""
    if not cfg.export_onnx:
        return None

    _ensure_dir()
    model_path = model_path or CHECKPOINT_DIR / "final_model.keras"

    try:
        import onnx  # noqa: F401
        import tf2onnx
    except ImportError:
        print("[export] tf2onnx and onnx are not installed. Skipping ONNX export.")
        return None

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first or pass model_path."
        )

    model = tf.keras.models.load_model(str(model_path))
    onnx_path = EXPORT_DIR / "model.onnx"

    if cfg.use_hybrid:
        input_signature = [
            tf.TensorSpec(
                shape=(1, *cfg.image_size, 3),
                dtype=tf.float32,
                name="image_input",
            ),
            tf.TensorSpec(
                shape=(1, len(TABULAR_FEATURES)),
                dtype=tf.float32,
                name="tabular_input",
            ),
        ]
    else:
        input_signature = [
            tf.TensorSpec(
                shape=(1, *cfg.image_size, 3),
                dtype=tf.float32,
                name="image_input",
            ),
        ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        output_path=str(onnx_path),
    )
    print(f"[export] ONNX saved: {onnx_path} ({_model_size_mb(onnx_path):.2f} MB)")
    return onnx_path


def benchmark_tflite(
    tflite_path: Path,
    cfg: TrainConfig,
    num_runs: int = 100,
) -> float:
    """Measure average CPU inference latency for a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_inputs = {
        detail["index"]: np.random.random(detail["shape"]).astype(detail["dtype"])
        for detail in input_details
    }

    for index, data in dummy_inputs.items():
        interpreter.set_tensor(index, data)
    interpreter.invoke()

    times_ms: list[float] = []
    for _ in range(num_runs):
        for index, data in dummy_inputs.items():
            interpreter.set_tensor(index, data)
        start = time.perf_counter()
        interpreter.invoke()
        times_ms.append((time.perf_counter() - start) * 1000)

    average_ms = float(np.mean(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    print(f"[benchmark] TFLite latency over {num_runs} runs:")
    print(f"  avg = {average_ms:.2f} ms | p95 = {p95_ms:.2f} ms")
    print(f"  output tensors = {len(output_details)}")
    print(f"  model size = {_model_size_mb(tflite_path):.2f} MB")
    return average_ms
