"""System status and metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse

from pig_behavior.api.dashboard import DASHBOARD_HTML
from pig_behavior.api.dependencies import PTModelService, get_model_service
from pig_behavior.api.schemas import HealthResponse, MetadataResponse
from pig_behavior.config import (
    BEHAVIOR_SEQUENCE_FEATURES,
    BEHAVIOR_SEQUENCE_LABELS,
    TABULAR_FEATURES,
)

router = APIRouter()


@router.get("/", response_model=MetadataResponse)
def root(model_service=Depends(get_model_service)) -> MetadataResponse:
    """Return API metadata and supported features."""
    return _metadata_response(model_service)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Return the HTML dashboard interface."""
    return DASHBOARD_HTML


@router.get("/health", response_model=HealthResponse)
def health(model_service=Depends(get_model_service)) -> HealthResponse:
    """Return basic health check status."""
    model_path = model_service.model_path
    return HealthResponse(
        status="ok",
        model_available=model_path.exists(),
        model_loaded=model_service.model_loaded,
        model_path=str(model_path),
    )


@router.get("/ready", response_model=HealthResponse)
def ready(model_service=Depends(get_model_service)) -> HealthResponse:
    """Load the model and return readiness status."""
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


@router.get("/metadata", response_model=MetadataResponse)
def metadata(model_service=Depends(get_model_service)) -> MetadataResponse:
    """Return explicit metadata about the loaded model."""
    return _metadata_response(model_service)


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
