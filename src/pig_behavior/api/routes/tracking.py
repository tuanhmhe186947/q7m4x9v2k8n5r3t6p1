"""Video tracking and streaming endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from pig_behavior.api.dependencies import get_tracking_session
from pig_behavior.api.schemas import TrackingStartRequest
from pig_behavior.config import (
    BEHAVIOR_CLASSIFIER_WEIGHTS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_VIDEO_PATH,
)
from pig_behavior.services.video_tracking import TrackingConfig

router = APIRouter(prefix="/tracking")


@router.post("/start")
def start_tracking(
    request: TrackingStartRequest,
    tracking_session=Depends(get_tracking_session),
) -> dict[str, object]:
    """Start the background tracking session."""
    try:
        config = _tracking_config_from_request(request)
        tracking_session.start(config)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return tracking_session.snapshot()


@router.post("/stop")
def stop_tracking(
    tracking_session=Depends(get_tracking_session),
) -> dict[str, object]:
    """Stop the background tracking session."""
    tracking_session.stop()
    return tracking_session.snapshot()


@router.get("/status")
def tracking_status(
    tracking_session=Depends(get_tracking_session),
) -> dict[str, object]:
    """Get the status of the background tracking session."""
    return tracking_session.snapshot()


@router.get("/stream")
def tracking_stream(
    tracking_session=Depends(get_tracking_session),
) -> StreamingResponse:
    """Stream MJPEG frames from the tracking session."""
    return StreamingResponse(
        tracking_session.frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
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
