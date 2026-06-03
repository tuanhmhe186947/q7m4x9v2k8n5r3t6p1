"""FastAPI service for pig behavior inference."""

from __future__ import annotations

import json
import math
import os
import re
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from pig_behavior import __version__
from pig_behavior.api.dashboard import DASHBOARD_HTML
from pig_behavior.api.schemas import (
    DetectionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    PredictionScore,
    TrackingStartRequest,
)
from pig_behavior.config import (
    BEHAVIOR_CLASSIFIER_WEIGHTS,
    BEHAVIOR_SEQUENCE_FEATURES,
    BEHAVIOR_SEQUENCE_LABELS,
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
from pig_behavior.services.pt_inference import PTModelService, PTPrediction
from pig_behavior.services.video_tracking import TrackingConfig, VideoTrackingSession


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


def create_app(service: TFLiteModelService | PTModelService | None = None) -> FastAPI:
    """Create the FastAPI application."""
    model_service = service or _service_from_env()
    tracking_session = VideoTrackingSession(_tracking_config_from_env())

    app = FastAPI(
        title="Pig Behavior API",
        version=__version__,
        summary="Inference API for the pig behavior classifier.",
    )

    @app.get("/", response_model=MetadataResponse)
    def root() -> MetadataResponse:
        return _metadata_response(model_service)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model_path = model_service.model_path
        return HealthResponse(
            status="ok",
            model_available=model_path.exists(),
            model_loaded=model_service.model_loaded,
            model_path=str(model_path),
        )

    @app.get("/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        try:
            model_service.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        model_path = getattr(model_service, "loaded_model_path", None)
        model_path = model_path or model_service.model_path
        return HealthResponse(
            status="ready",
            model_available=True,
            model_loaded=True,
            model_path=str(model_path),
        )

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        return _metadata_response(model_service)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_upload(
        image: Annotated[
            UploadFile,
            File(description="Pig image file, for example JPEG or PNG."),
        ],
        bbox: Annotated[
            str | None,
            Form(
                description=(
                    "Optional bounding box as JSON array or comma-separated "
                    "x1,y1,x2,y2 values."
                ),
            ),
        ] = None,
        tabular: Annotated[
            str | None,
            Form(
                description=(
                    "Optional six tabular values as JSON array or comma-separated "
                    "values in API metadata order."
                ),
            ),
        ] = None,
    ) -> PredictionResponse:
        content_type = image.content_type or ""
        if content_type and not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload must be an image.")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        parsed_bbox = _parse_float_tuple(
            bbox,
            expected_count=4,
            field_name="bbox",
        )
        parsed_tabular = _parse_float_list(
            tabular,
            expected_count=len(TABULAR_FEATURES),
            field_name="tabular",
        )

        try:
            if isinstance(model_service, PTModelService):
                pt_prediction = model_service.predict_bytes(
                    image_bytes,
                    bbox=parsed_bbox,
                    confidence_threshold=_confidence_from_env(),
                )
                return _pt_prediction_response(
                    image.filename,
                    model_service,
                    pt_prediction,
                    parsed_bbox,
                )

            scores, latency_ms = model_service.predict_image(
                image_bytes,
                parsed_bbox,
                parsed_tabular,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted_label, confidence = sorted_scores[0]
        model_path = model_service.loaded_model_path or model_service.model_path

        return PredictionResponse(
            filename=image.filename,
            backend="tflite",
            task="classify",
            predicted_label=predicted_label,
            confidence=confidence,
            latency_ms=latency_ms,
            scores=[
                PredictionScore(label=label, confidence=score)
                for label, score in sorted_scores
            ],
            detections=[],
            bbox=list(parsed_bbox) if parsed_bbox is not None else None,
            tabular=_tabular_dict(parsed_tabular),
            model_path=str(model_path),
        )

    @app.post("/tracking/start")
    def start_tracking(request: TrackingStartRequest) -> dict[str, object]:
        try:
            config = _tracking_config_from_request(request)
            tracking_session.start(config)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return tracking_session.snapshot()

    @app.post("/tracking/stop")
    def stop_tracking() -> dict[str, object]:
        tracking_session.stop()
        return tracking_session.snapshot()

    @app.get("/tracking/status")
    def tracking_status() -> dict[str, object]:
        return tracking_session.snapshot()

    @app.get("/tracking/stream")
    def tracking_stream() -> StreamingResponse:
        return StreamingResponse(
            tracking_session.frame_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def _metadata_response(service) -> MetadataResponse:
    if isinstance(service, PTModelService):
        return MetadataResponse(
            backend="pt",
            labels=BEHAVIOR_SEQUENCE_LABELS,
            tabular_features=BEHAVIOR_SEQUENCE_FEATURES,
            image_size=(224, 224),
            use_hybrid=False,
            use_coarse_labels=False,
            model_path=str(service.model_path),
        )

    return MetadataResponse(
        backend="tflite",
        labels=service.cfg.labels,
        tabular_features=TABULAR_FEATURES,
        image_size=service.cfg.image_size,
        use_hybrid=service.cfg.use_hybrid,
        use_coarse_labels=service.cfg.use_coarse_labels,
        model_path=str(service.model_path),
    )


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


def _tracking_config_from_request(request: TrackingStartRequest) -> TrackingConfig:
    detector_model_path = request.detector_model_path or request.model_path
    return TrackingConfig(
        detector_model_path=(
            Path(detector_model_path)
            if detector_model_path
            else DEFAULT_DETECTOR_MODEL
        ),
        behavior_model_path=(
            Path(request.behavior_model_path)
            if request.behavior_model_path
            else BEHAVIOR_CLASSIFIER_WEIGHTS
        ),
        video_path=(
            Path(request.video_path) if request.video_path else DEFAULT_VIDEO_PATH
        ),
        confidence=request.confidence,
        frame_stride=request.frame_stride,
        behavior_stride_frames=request.behavior_stride_frames,
        realtime=request.realtime,
    )


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


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_tuple(
    raw: str | None,
    *,
    expected_count: int,
    field_name: str,
) -> tuple[float, ...] | None:
    values = _parse_float_list(
        raw,
        expected_count=expected_count,
        field_name=field_name,
    )
    return tuple(values) if values is not None else None


def _parse_float_list(
    raw: str | None,
    *,
    expected_count: int,
    field_name: str,
) -> list[float] | None:
    if raw is None or not raw.strip():
        return None

    try:
        if raw.strip().startswith("["):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError
            values = [float(value) for value in parsed]
        else:
            tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
            values = [float(token) for token in tokens]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must contain numeric values.",
        ) from exc

    if len(values) != expected_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} must contain {expected_count} values, "
                f"got {len(values)}."
            ),
        )
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} values must be finite numbers.",
        )

    return values


def _tabular_dict(values: list[float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    return dict(zip(TABULAR_FEATURES, values, strict=True))


def _pt_prediction_response(
    filename: str | None,
    service: PTModelService,
    prediction: PTPrediction,
    bbox: tuple[float, ...] | None,
) -> PredictionResponse:
    scores = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
    return PredictionResponse(
        filename=filename,
        backend="pt",
        task=prediction.task,
        predicted_label=prediction.predicted_label or "",
        confidence=prediction.confidence or 0.0,
        latency_ms=prediction.latency_ms,
        scores=[
            PredictionScore(label=label, confidence=confidence)
            for label, confidence in scores
        ],
        detections=[
            DetectionResponse(
                label=detection.label,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
                class_id=detection.class_id,
            )
            for detection in prediction.detections
        ],
        bbox=list(bbox) if bbox is not None else None,
        tabular=None,
        model_path=str(service.model_path),
    )


def run() -> None:
    """Run the API with Uvicorn."""
    import uvicorn

    host = os.getenv("PIG_BEHAVIOR_API_HOST", "0.0.0.0")
    port = int(os.getenv("PIG_BEHAVIOR_API_PORT", "8000"))
    uvicorn.run("pig_behavior.api:app", host=host, port=port)


app = create_app()


if __name__ == "__main__":
    run()
