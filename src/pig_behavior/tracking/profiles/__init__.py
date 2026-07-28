"""Named tracking profiles used by tracking/evaluation scripts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType

from pig_behavior.tracking.profiles.bytetrack_raw import (
    EVAL_CONFIGS as BYTETRACK_EVAL_CONFIGS,
)
from pig_behavior.tracking.profiles.bytetrack_raw import (
    PRESENTATION_PROFILES as BYTETRACK_PRESENTATION_PROFILES,
)
from pig_behavior.tracking.profiles.hybrid_bytetrack import (
    EVAL_CONFIGS as HYBRID_EVAL_CONFIGS,
)
from pig_behavior.tracking.profiles.hybrid_bytetrack import (
    HYBRID_BASE_CONFIG,
)
from pig_behavior.tracking.profiles.hybrid_bytetrack import (
    PRESENTATION_PROFILES as HYBRID_PRESENTATION_PROFILES,
)
from pig_behavior.tracking.profiles.realtime import (
    EVAL_CONFIGS as REALTIME_EVAL_CONFIGS,
)
from pig_behavior.tracking.profiles.realtime import (
    PRESENTATION_PROFILES as REALTIME_PRESENTATION_PROFILES,
)
from pig_behavior.tracking.profiles.realtime import REALTIME_BASE_CONFIG
from pig_behavior.tracking.profiles.rf_hybrid_offline import (
    EVAL_CONFIGS as RF_HYBRID_EVAL_CONFIGS,
)
from pig_behavior.tracking.profiles.rf_hybrid_offline import (
    PRESENTATION_PROFILES as RF_HYBRID_PRESENTATION_PROFILES,
)

RULE_BENCHMARK_OVERRIDE_KEYS = {
    "USE_IOU_FALLBACK",
    "USE_AREA_OCCLUSION_FREEZE",
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE",
    "USE_MERGED_BOX_SPLIT",
}

EVAL_CONFIG_OVERRIDES: Mapping[str, dict[str, object]] = MappingProxyType(
    {
        **HYBRID_EVAL_CONFIGS,
        **BYTETRACK_EVAL_CONFIGS,
        **REALTIME_EVAL_CONFIGS,
        **RF_HYBRID_EVAL_CONFIGS,
    }
)

PRESENTATION_PROFILES: Mapping[str, dict[str, object]] = MappingProxyType(
    {
        **BYTETRACK_PRESENTATION_PROFILES,
        **REALTIME_PRESENTATION_PROFILES,
        **HYBRID_PRESENTATION_PROFILES,
        **RF_HYBRID_PRESENTATION_PROFILES,
    }
)

BASE_EVAL_CONFIG = HYBRID_BASE_CONFIG
REALTIME_EVAL_CONFIG = REALTIME_BASE_CONFIG

RETIRED_PROFILE_MESSAGES: Mapping[str, str] = MappingProxyType(
    {
        "realtime": (
            "Profile 'realtime' has been retired. Use 'realtime_fast'."
        ),
        "realtime_balanced": (
            "Profile 'realtime_balanced' is historical and unavailable "
            "for active execution."
        ),
        "realtime_quality_delayed": (
            "Profile 'realtime_quality_delayed' is historical and "
            "unavailable for active execution."
        ),
        "realtime_fast_h1_r2": (
            "Profile 'realtime_fast_h1_r2' is a rejected experimental "
            "profile and unavailable for active execution."
        ),
    }
)


class RetiredTrackingProfileError(ValueError):
    """Raised when historical profile data is requested for execution."""


def _raise_if_retired(name: str) -> None:
    message = RETIRED_PROFILE_MESSAGES.get(name)
    if message is not None:
        raise RetiredTrackingProfileError(message)


def get_eval_config(name: str) -> dict[str, object]:
    """Return a mutable copy of a named TrackingConfig override set."""

    _raise_if_retired(name)
    return deepcopy(EVAL_CONFIG_OVERRIDES[name])


def get_presentation_profile(name: str) -> dict[str, object]:
    """Return a mutable copy of a presentation-oriented tracking profile."""

    _raise_if_retired(name)
    return deepcopy(PRESENTATION_PROFILES[name])


def format_profile_override_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
