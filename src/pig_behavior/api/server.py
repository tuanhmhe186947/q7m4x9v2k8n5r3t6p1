"""FastAPI application factory and server entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI

from pig_behavior import __version__
from pig_behavior.api.dependencies import (
    _service_from_env,
    _tracking_config_from_env,
)
from pig_behavior.api.routes import predict, system, tracking
from pig_behavior.services.pt_inference import PTModelService
from pig_behavior.api.dependencies import TFLiteModelService
from pig_behavior.services.video_tracking import VideoTrackingSession


def create_app(service: TFLiteModelService | PTModelService | None = None) -> FastAPI:
    """Create the FastAPI application."""
    model_service = service or _service_from_env()
    tracking_session = VideoTrackingSession(_tracking_config_from_env())

    app = FastAPI(
        title="Pig Behavior API",
        version=__version__,
        summary="Inference API for the pig behavior classifier.",
    )

    # Attach to state for dependencies
    app.state.model_service = model_service
    app.state.tracking_session = tracking_session

    # Include routes
    app.include_router(system.router)
    app.include_router(predict.router)
    app.include_router(tracking.router)

    return app


def run() -> None:
    """Run the API with Uvicorn."""
    import uvicorn

    host = os.getenv("PIG_BEHAVIOR_API_HOST", "0.0.0.0")
    port = int(os.getenv("PIG_BEHAVIOR_API_PORT", "8000"))
    uvicorn.run("pig_behavior.api.server:app", host=host, port=port)


# The default app instance for Uvicorn
app = create_app()
