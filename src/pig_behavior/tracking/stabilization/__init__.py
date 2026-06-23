"""Stable annotation tracking package."""

from __future__ import annotations

from pig_behavior.tracking.stabilization.config import AnnotationStableConfig
from pig_behavior.tracking.stabilization.stable_tracker import run_stable_tracking

__all__ = [
    "AnnotationStableConfig",
    "run_stable_tracking",
]
