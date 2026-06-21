"""FastAPI service for pig behavior inference."""

from __future__ import annotations

from pig_behavior.api.server import (
    TFLiteModelService,
    app,
    create_app,
    run,
)

__all__ = [
    "TFLiteModelService",
    "app",
    "create_app",
    "run",
]
