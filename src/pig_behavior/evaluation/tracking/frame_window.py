"""Inclusive frame-window validation for tracking evaluation."""

from __future__ import annotations


def _optional_nonnegative_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None.")


def validate_frame_bounds(
    start_frame: int | None,
    end_frame: int | None,
) -> None:
    """Validate optional inclusive evaluation bounds."""

    _optional_nonnegative_int("evaluation_start_frame", start_frame)
    _optional_nonnegative_int("evaluation_end_frame", end_frame)
    if (
        start_frame is not None
        and end_frame is not None
        and end_frame < start_frame
    ):
        raise ValueError(
            "evaluation_end_frame must be greater than or equal to "
            "evaluation_start_frame."
        )


def validate_generated_frame_coverage(
    *,
    tracking_start_frame: int,
    max_frames: int | None,
    evaluation_start_frame: int | None,
    evaluation_end_frame: int | None,
) -> None:
    """Require a generated prediction interval to cover its scored interval."""

    validate_frame_bounds(evaluation_start_frame, evaluation_end_frame)
    _optional_nonnegative_int("tracking_start_frame", tracking_start_frame)
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            raise ValueError("max_frames must be a positive integer or None.")
        if max_frames < 1:
            raise ValueError("max_frames must be a positive integer or None.")

    if (
        evaluation_start_frame is not None
        and tracking_start_frame > evaluation_start_frame
    ):
        raise ValueError(
            "Generated tracking must start at or before evaluation_start_frame."
        )

    if evaluation_end_frame is not None and max_frames is not None:
        generated_end_frame = tracking_start_frame + max_frames - 1
        if generated_end_frame < evaluation_end_frame:
            raise ValueError(
                "Generated prediction interval does not cover "
                "evaluation_end_frame."
            )


def frame_is_in_bounds(
    frame: int,
    start_frame: int | None,
    end_frame: int | None,
) -> bool:
    """Return whether one frame is inside optional inclusive bounds."""

    if start_frame is not None and frame < start_frame:
        return False
    return end_frame is None or frame <= end_frame


__all__ = [
    "frame_is_in_bounds",
    "validate_frame_bounds",
    "validate_generated_frame_coverage",
]
