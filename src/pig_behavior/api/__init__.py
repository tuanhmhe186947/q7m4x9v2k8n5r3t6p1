"""FastAPI package entrypoint.

This module preserves ``uvicorn pig_behavior.api:app`` while the implementation
lives in ``pig_behavior.api.app``.
"""

from pig_behavior.api.app import app, create_app, run

__all__ = ["app", "create_app", "run"]
