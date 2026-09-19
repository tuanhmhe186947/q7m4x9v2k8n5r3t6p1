"""Canonical RF hybrid-mechanism transfer profile."""

from __future__ import annotations

from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG

RF_HYBRID_CONFIG: dict[str, object] = {
    **REALTIME_FAST_CONFIG,
    "rf_hybrid_transfer": True,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "rf_hybrid": RF_HYBRID_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "rf_hybrid": {
        "mode": "realtime",
        "eval_config": "rf_hybrid",
        "description": (
            "Frozen realtime_fast tracklets followed by the predeclared "
            "portable hybrid-mechanism subset."
        ),
    },
}
