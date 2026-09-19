"""Shared temporal-input tier names for legacy classifier ablations."""

from __future__ import annotations

DEFAULT_TEMPORAL_TIERS = (6, 8, 12, 16)
TEMPORAL_TIER_VIEWS = (
    "all_sliding_event_balanced",
    "one_centered_window_matched",
)


def legacy_model_view_name(length: int, sampling_view: str) -> str:
    """Name one model input while encoding length and sampling protocol."""

    if length not in DEFAULT_TEMPORAL_TIERS:
        raise ValueError(f"unsupported legacy temporal tier={length}")
    if sampling_view not in TEMPORAL_TIER_VIEWS:
        raise ValueError(f"unsupported legacy sampling view={sampling_view}")
    suffix = (
        "all_sliding"
        if sampling_view == "all_sliding_event_balanced"
        else "centered_matched"
    )
    return f"legacy_t{length}_{suffix}_observed_time"


LEGACY_TEMPORAL_MODEL_VIEW_SPECS = {
    legacy_model_view_name(length, sampling_view): {
        "sequence_length": length,
        "sampling_view": sampling_view,
        "selection_column": (
            f"legacy_t{length}_"
            + (
                "all_sliding_keep"
                if sampling_view == "all_sliding_event_balanced"
                else "centered_matched_keep"
            )
        ),
        "slot_manifest_filename": (
            f"legacy_t{length}_"
            + (
                "all_sliding_observed_time_manifest.csv"
                if sampling_view == "all_sliding_event_balanced"
                else "centered_matched_observed_time_manifest.csv"
            )
        ),
    }
    for length in DEFAULT_TEMPORAL_TIERS
    for sampling_view in TEMPORAL_TIER_VIEWS
}

__all__ = [
    "DEFAULT_TEMPORAL_TIERS",
    "LEGACY_TEMPORAL_MODEL_VIEW_SPECS",
    "TEMPORAL_TIER_VIEWS",
    "legacy_model_view_name",
]
