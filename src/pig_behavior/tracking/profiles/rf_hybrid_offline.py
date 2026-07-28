"""Development-only RF core plus frozen offline-repair profile."""

from __future__ import annotations

from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG

RF_HYBRID_OFFLINE_CONFIG: dict[str, object] = {
    **REALTIME_FAST_CONFIG,
    "rf_hybrid_offline": True,
    "write_output_video": False,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "rf_hybrid_offline": RF_HYBRID_OFFLINE_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "rf_hybrid_offline": {
        "mode": "realtime",
        "eval_config": "rf_hybrid_offline",
        "description": (
            "Development-only realtime_fast raw core followed by the frozen "
            "post-video offline repair stack."
        ),
        "role": "development_only_offline_repair",
        "realtime_core": "realtime_fast",
        "timing_class": "post_video_offline",
        "causal_zero_delay": False,
        "output_delay_frames": -1,
        "future_frames_may_be_used": True,
        "production_default": False,
        "promotion_status": "NOT_PROMOTED",
        "unseen_execution_authorized": False,
    },
}

