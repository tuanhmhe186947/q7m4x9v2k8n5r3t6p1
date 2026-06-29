"""Classification dataset pipeline v2.

This package builds a unified spatio-temporal behavior dataset from:
- legacy recovered burst annotations,
- CVAT tracking XML annotations,
- CVAT selected/native behavior annotations.

The v2 pipeline keeps the old classification pipeline intact and introduces
a canonical frame-object schema for richer geometry, motion, ROI, and social
features.
"""

from __future__ import annotations

from .schema import (
    ANNOTATION_SCOPES,
    BEHAVIOR_TO_COARSE,
    CANONICAL_FRAME_OBJECT_COLUMNS,
    DEFAULT_PIG_IDS,
    INTERACTION_BEHAVIORS,
    MOTION_DOMINANT_BEHAVIORS,
    ROI_DOMINANT_BEHAVIORS,
    SHAPE_DOMINANT_BEHAVIORS,
    SOURCE_TYPES,
    VALID_BEHAVIORS,
    behavior_to_coarse,
    is_interaction_behavior,
    normalize_behavior,
    normalize_hidden,
)

__all__ = [
    "ANNOTATION_SCOPES",
    "BEHAVIOR_TO_COARSE",
    "CANONICAL_FRAME_OBJECT_COLUMNS",
    "DEFAULT_PIG_IDS",
    "INTERACTION_BEHAVIORS",
    "MOTION_DOMINANT_BEHAVIORS",
    "ROI_DOMINANT_BEHAVIORS",
    "SHAPE_DOMINANT_BEHAVIORS",
    "SOURCE_TYPES",
    "VALID_BEHAVIORS",
    "behavior_to_coarse",
    "is_interaction_behavior",
    "normalize_behavior",
    "normalize_hidden",
]