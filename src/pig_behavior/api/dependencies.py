"""FastAPI dependencies and service singletons."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image
from io import BytesIO

from pig_behavior.config import (
    BEHAVIOR_CLASSIFIER_WEIGHTS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_VIDEO_PATH,
    TABULAR_FEATURES,
    TrainConfig,
)
from pig_behavior.inference import (
    load_interpreter,
    predict,
    preprocess_pil_image,
    resolve_tflite_path,
)
from pig_behavior.services.pt_inference import PTModelService
from pig_behavior.services.video_tracking import TrackingConfig


class TFLiteModelService:
    """Thread-safe wrapper around a single TFLite interpreter."""

    def __init__(
        self,
        cfg: TrainConfig,
        tflite_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self._configured_tflite_path = tflite_path
        self._interpreter = None
        self._loaded_model_path: Path | None = None
        self._lock = Lock()

    @property
    def model_path(self) -> Path:
        """Return the configured or default TFLite model path."""
        return resolve_tflite_path(self._configured_tflite_path)

    @property
    def model_loaded(self) -> bool:
        """Return whether the interpreter has already been loaded."""
        return self._interpreter is not None

    @property
    def loaded_model_path(self) -> Path | None:
        """Return the loaded model path when available."""
        return self._loaded_model_path

    @property
    def model_available(self) -> bool:
        """Return whether the resolved model path exists on disk."""
        return self.model_path.exists()

    def load(self) -> None:
        """Load the interpreter if it has not been loaded yet."""
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        if self._interpreter is None:
            model_path = self.model_path
            self._interpreter = load_interpreter(model_path)
            self._loaded_model_path = model_path
        return self._interpreter

    def predict_image(
        self,
        image_bytes: bytes,
        bbox: tuple[float, float, float, float] | None,
        tabular_features: list[float] | None,
    ) -> tuple[dict[str, float], float]:
        """Run one image prediction."""
        with Image.open(BytesIO(image_bytes)) as image:
            image_array = preprocess_pil_image(image, bbox, self.cfg.image_size)

        tabular_array = None
        if tabular_features is not None:
            if len(tabular_features) != len(TABULAR_FEATURES):
                raise ValueError(
                    f"Expected {len(TABULAR_FEATURES)} tabular values, "
                    f"got {len(tabular_features)}."
                )
            tabular_array = np.asarray([tabular_features], dtype=np.float32)
        elif self.cfg.use_hybrid:
            raise ValueError(
                "The hybrid model expects tabular features. "
                "Provide six values with the tabular form field."
            )

        with self._lock:
            interpreter = self._load_unlocked()
            return predict(
                interpreter,
                image_array,
                tabular_array,
                self.cfg.labels,
            )


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_from_env() -> TrainConfig:
    return TrainConfig(
        use_hybrid=not _env_flag("PIG_BEHAVIOR_IMAGE_ONLY", default=False),
        use_coarse_labels=_env_flag("PIG_BEHAVIOR_COARSE_LABELS", default=False),
    )


def _model_path_from_env() -> Path | None:
    value = os.getenv("PIG_BEHAVIOR_TFLITE_PATH")
    return Path(value) if value else None


def _pt_model_path_from_env() -> Path:
    value = os.getenv("PIG_BEHAVIOR_PT_MODEL_PATH")
    return Path(value) if value else BEHAVIOR_CLASSIFIER_WEIGHTS


def _confidence_from_env() -> float:
    return float(os.getenv("PIG_BEHAVIOR_CONF", "0.25"))


def _service_from_env():
    backend = os.getenv("PIG_BEHAVIOR_MODEL_BACKEND", "auto").strip().lower()
    pt_path = _pt_model_path_from_env()

    if backend not in {"auto", "tflite", "pt"}:
        raise ValueError(
            "PIG_BEHAVIOR_MODEL_BACKEND must be one of: auto, tflite, pt."
        )
    if backend == "pt" or (backend == "auto" and pt_path.exists()):
        return PTModelService(pt_path)
    return TFLiteModelService(_config_from_env(), _model_path_from_env())


def _tracking_config_from_env() -> TrackingConfig:
    detector_model_path = os.getenv("PIG_BEHAVIOR_DETECT_MODEL_PATH")
    behavior_model_path = os.getenv("PIG_BEHAVIOR_PT_MODEL_PATH")
    video_path = os.getenv("PIG_BEHAVIOR_VIDEO_PATH")
    return TrackingConfig(
        detector_model_path=(
            Path(detector_model_path)
            if detector_model_path
            else DEFAULT_DETECTOR_MODEL
        ),
        behavior_model_path=(
            Path(behavior_model_path)
            if behavior_model_path
            else BEHAVIOR_CLASSIFIER_WEIGHTS
        ),
        video_path=Path(video_path) if video_path else DEFAULT_VIDEO_PATH,
        confidence=_confidence_from_env(),
        frame_stride=max(1, int(os.getenv("PIG_BEHAVIOR_FRAME_STRIDE", "1"))),
        behavior_stride_frames=max(
            1,
            int(os.getenv("PIG_BEHAVIOR_BEHAVIOR_STRIDE_FRAMES", "3")),
        ),
        realtime=_env_flag("PIG_BEHAVIOR_REALTIME", default=True),
    )


# FastAPI Dependencies
from fastapi import Request

def get_model_service(request: Request):
    """Retrieve the model service from the app state."""
    return request.app.state.model_service

def get_tracking_session(request: Request):
    """Retrieve the tracking session from the app state."""
    return request.app.state.tracking_session
