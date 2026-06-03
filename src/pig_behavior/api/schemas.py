"""FastAPI request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health without forcing model loading."""

    status: str
    model_available: bool
    model_loaded: bool
    model_path: str


class MetadataResponse(BaseModel):
    """Runtime model contract exposed to clients."""

    backend: str
    labels: list[str]
    tabular_features: list[str]
    image_size: tuple[int, int] | None
    use_hybrid: bool
    use_coarse_labels: bool
    model_path: str


class PredictionScore(BaseModel):
    """One class confidence score."""

    label: str
    confidence: float = Field(ge=0.0)


class DetectionResponse(BaseModel):
    """One detection returned by a detector model."""

    label: str
    confidence: float = Field(ge=0.0)
    bbox_xyxy: list[float]
    class_id: int


class PredictionResponse(BaseModel):
    """Prediction result returned by the API."""

    filename: str | None
    backend: str
    task: str
    predicted_label: str
    confidence: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    scores: list[PredictionScore]
    detections: list[DetectionResponse]
    bbox: list[float] | None
    tabular: dict[str, float] | None
    model_path: str


class TrackingStartRequest(BaseModel):
    """Request body for starting real-time video tracking."""

    detector_model_path: str | None = None
    behavior_model_path: str | None = None
    model_path: str | None = None
    video_path: str | None = None
    confidence: float = Field(default=0.25, gt=0.0, lt=1.0)
    frame_stride: int = Field(default=1, ge=1, le=120)
    behavior_stride_frames: int = Field(default=3, ge=1, le=60)
    realtime: bool = True
