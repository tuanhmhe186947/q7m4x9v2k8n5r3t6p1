"""Video annotation rendering helpers for fixed-ID pig tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.constants import TRACK_COLORS_BGR
from pig_behavior.tracking.masks import load_mask, shade_outside_roi
from pig_behavior.tracking.refinement import shape_box, shape_hidden_value
from pig_behavior.tracking.schemas import FixedTrack
from pig_behavior.tracking.tracks import track_is_hidden


def draw_dashed_rectangle(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 10,
) -> None:
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_tracks(
    frame: np.ndarray,
    tracks: dict[int, FixedTrack],
    mask: np.ndarray | None,
    frame_index: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    import cv2

    vis = shade_outside_roi(frame, mask) if cfg.shade_outside_mask else frame.copy()
    overlay = vis.copy()
    if mask is not None and cfg.draw_mask_outline:
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    cv2.putText(
        overlay,
        f"frame {frame_index}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for fixed_id in range(1, cfg.expected_pigs + 1):
        track = tracks[fixed_id]
        x1, y1, x2, y2 = track.last_box.astype(int)
        color = TRACK_COLORS_BGR[fixed_id]
        hidden = track_is_hidden(track, cfg)
        if hidden:
            draw_dashed_rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"Pig_{fixed_id} / ID_{fixed_id}"
        if hidden:
            label += " hidden"
        elif track.last_score < cfg.review_conf:
            label += " review"
        if track.last_ambiguous:
            label += " occ"
        if track.last_source == "occlusion_hold":
            label += " hold"
        if hidden or track.last_ambiguous or track.last_source == "occlusion_hold":
            state_label = {
                "moving": "move",
                "stationary": "stay",
                "unknown": "unk",
            }.get(track.motion_state, "unk")
            label += f" {state_label}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    alpha = float(np.clip(cfg.visual_opacity, 0.0, 1.0))
    return cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0)


def draw_shape_annotations(
    frame: np.ndarray,
    shapes: list[dict[str, Any]],
    mask: np.ndarray | None,
    frame_index: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    import cv2

    vis = shade_outside_roi(frame, mask) if cfg.shade_outside_mask else frame.copy()
    overlay = vis.copy()
    if mask is not None and cfg.draw_mask_outline:
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    cv2.putText(
        overlay,
        f"frame {frame_index}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    ordered_shapes = sorted(
        shapes,
        key=lambda item: int(str(item["label"]).removeprefix("Pig_")),
    )
    for shape in ordered_shapes:
        fixed_id = int(str(shape["label"]).removeprefix("Pig_"))
        x1, y1, x2, y2 = shape_box(shape).astype(int)
        color = TRACK_COLORS_BGR[fixed_id]
        hidden = shape_hidden_value(shape) == "Yes"
        if hidden:
            draw_dashed_rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"Pig_{fixed_id} / ID_{fixed_id}"
        if hidden:
            label += " hidden"
        elif shape.get("_needs_review"):
            label += " review"
        if shape.get("_refined"):
            label += " refined"
        if shape.get("_ambiguous_occlusion"):
            label += " occ"
        if shape.get("_occlusion_hold"):
            label += " hold"
        if hidden or shape.get("_ambiguous_occlusion") or shape.get("_occlusion_hold"):
            state_label = {
                "moving": "move",
                "stationary": "stay",
                "unknown": "unk",
            }.get(str(shape.get("_motion_state", "unknown")), "unk")
            label += f" {state_label}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    alpha = float(np.clip(cfg.visual_opacity, 0.0, 1.0))
    return cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0)


def shapes_by_frame(shapes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes:
        grouped.setdefault(int(shape["frame"]), []).append(shape)
    return grouped


def render_annotation_video(
    video_path: Path,
    output_video: Path,
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
    frame_limit: int | None = None,
) -> int:
    """Render final preview video from refined annotation shapes."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for rendering: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video frame size for rendering.")
    if cfg.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    mask = load_mask(cfg.mask_path, width, height, cfg)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    grouped_shapes = shapes_by_frame(shapes)
    frames_rendered = 0
    try:
        while True:
            if frame_limit is not None and frames_rendered >= frame_limit:
                break
            if cfg.max_frames is not None and frames_rendered >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_h, frame_w = frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(cfg.mask_path, width, height, cfg)
            frame_index = cfg.start_frame + frames_rendered
            annotated = draw_shape_annotations(
                frame,
                grouped_shapes.get(frame_index, []),
                mask,
                frame_index,
                cfg,
            )
            writer.write(annotated)
            frames_rendered += 1
    finally:
        capture.release()
        writer.release()

    return frames_rendered


__all__ = [
    "draw_dashed_rectangle",
    "draw_shape_annotations",
    "draw_tracks",
    "render_annotation_video",
    "shapes_by_frame",
]
