"""Track eight pigs in a 30 FPS video and export CVAT-style annotations.

Notebook quick start:

    cfg = TrackingConfig(
        video_path=Path("data/videos/pigs101219_full.mp4"),
        weights_path=Path("models/detector/pig_detector_yolo.pt"),
        mask_path=Path("data/annotations/scene/mask.png"),
    )
    summary = run_tracking(cfg)
    display_tracked_video(summary.output_video)

Command line:

    pig-track-for-annotation \
        --video data/videos/pigs101219_full.mp4 \
        --weights models/detector/pig_detector_yolo.pt \
        --mask data/annotations/scene/mask.png \
        --det-conf 0.25 \
        --track-high-conf 0.50 \
        --review-conf 0.75 \
        --iou 0.80 \
        --visual-opacity 0.75
"""

# ruff: noqa

# %%
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np

from pig_behavior.tracking_path_config import (
    DEFAULT_TRACKING_PATH_CONFIG,
    load_tracking_path_profile,
    profile_path,
    profile_video_path,
    profile_video_paths,
)


def _find_project_root(start: Path) -> Path:
    """Return the nearest parent containing the project metadata."""
    start = start if start.is_dir() else start.parent
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "Pigs281119_000085_30fps.mp4"
DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"
)
DEFAULT_MASK_PATH = PROJECT_ROOT / "data" / "annotations" / "scene" / "mask.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "id_tracking"

ID_VALUES = [f"ID_{idx}" for idx in range(1, 9)]
BEHAVIOR_VALUES = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]
TRACK_COLORS_BGR = {
    1: (65, 105, 225),
    2: (50, 205, 50),
    3: (255, 140, 0),
    4: (220, 20, 60),
    5: (148, 0, 211),
    6: (0, 206, 209),
    7: (255, 20, 147),
    8: (154, 205, 50),
}
DEFAULT_DET_CONF_THRESHOLD = 0.25
DEFAULT_TRACK_HIGH_CONF_THRESHOLD = 0.50
DEFAULT_REVIEW_CONF_THRESHOLD = 0.75
DEFAULT_CONF_THRESHOLD = DEFAULT_REVIEW_CONF_THRESHOLD
DEFAULT_OVERLAP_THRESHOLD = 0.80
DEFAULT_VISUAL_OPACITY = 0.75


def _bgr_to_hex(color: tuple[int, int, int]) -> str:
    blue, green, red = color
    return f"#{red:02X}{green:02X}{blue:02X}"


def build_pig_label_schema() -> list[dict[str, Any]]:
    """Build CVAT labels where Pig_N and attribute ID_N are locked together."""
    labels: list[dict[str, Any]] = []
    for idx, id_value in enumerate(ID_VALUES, start=1):
        attr_base_id = 2522600 + idx * 10
        labels.append(
            {
                "name": f"Pig_{idx}",
                "id": 7872368 + idx - 1,
                "color": _bgr_to_hex(TRACK_COLORS_BGR[idx]),
                "type": "any",
                "attributes": [
                    {
                        "id": attr_base_id + 1,
                        "name": "ID",
                        "input_type": "select",
                        "mutable": False,
                        "values": [id_value],
                        "default_value": id_value,
                    },
                    {
                        "id": attr_base_id + 2,
                        "name": "Behavior",
                        "input_type": "select",
                        "mutable": True,
                        "values": BEHAVIOR_VALUES,
                        "default_value": "lying",
                    },
                    {
                        "id": attr_base_id + 3,
                        "name": "Hidden",
                        "input_type": "select",
                        "mutable": True,
                        "values": ["No", "Yes"],
                        "default_value": "No",
                    },
                ],
            }
        )
    return labels


PIG_LABEL_SCHEMA = build_pig_label_schema()


# %%
@dataclass(slots=True)
class TrackingConfig:
    """Config for YOLOv8 detection, ID stabilization, JSON, and video export."""

    video_path: Path = DEFAULT_VIDEO_PATH
    weights_path: Path = DEFAULT_WEIGHTS_PATH
    mask_path: Path | None = DEFAULT_MASK_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_video: Path | None = None
    annotations_json: Path | None = None
    coco_annotations_json: Path | None = None
    clean_coco_annotations_json: Path | None = None
    cvat_video_xml: Path | None = None
    labels_json: Path | None = None
    tracker_yaml: Path | None = None
    quality_report_json: Path | None = None
    quality_report_csv: Path | None = None

    expected_pigs: int = 8
    output_fps: float = 30.0
    start_frame: int = 0
    det_conf: float = DEFAULT_DET_CONF_THRESHOLD
    track_high_conf: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    review_conf: float = DEFAULT_REVIEW_CONF_THRESHOLD
    adaptive_conf_step: float = 0.05
    conf: float | None = None
    iou: float = DEFAULT_OVERLAP_THRESHOLD
    class_id: int | None = None
    allowed_class_name: str | None = None

    use_mask: bool = True
    mask_input_frame: bool = True
    roi_mode: str = "center"
    roi_min_cover: float = 0.10
    roi_dilate_px: int = 8

    max_missing_frames: int = 90
    hidden_missed_frames: int = 5
    hidden_score_threshold: float = 0.15
    use_mask_iou: bool = True
    mask_iou_max_missed: int = 10
    mask_iou_min_area: int = 64
    match_cost_threshold: float = 0.78
    unseen_track_cost_threshold: float = 1.10
    lost_track_cost_threshold: float = 0.95
    lost_track_reid_appearance_threshold: float = 0.25
    duplicate_iou_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    initial_track_conf: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    low_conf_motion_gate: bool = True
    motion_gate_confidence: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    low_conf_max_center_jump: float = 0.08
    low_conf_max_box_jump_scale: float = 1.75
    low_conf_min_iou: float = 0.01
    occlusion_aware_matching: bool = True
    occlusion_track_iom_threshold: float = 0.20
    occlusion_detection_iom_threshold: float = 0.30
    occlusion_stationary_speed: float = 0.006
    occlusion_stationary_max_center_jump: float = 0.045
    occlusion_switch_penalty: float = 0.45
    occlusion_competitor_margin: float = 0.12
    occlusion_appearance_penalty: float = 0.30
    occlusion_appearance_margin: float = 0.08
    occlusion_stationary_lock: bool = True
    freeze_identity_in_occlusion: bool = True
    hold_occluded_box: bool = True
    occlusion_hold_max_frames: int = 30
    occlusion_hold_hidden_frames: int = 2
    identity_swap_guard: bool = True
    identity_swap_min_gain: float = 0.015
    identity_swap_iom_threshold: float = 0.10
    hidden_motion_model: bool = True
    hidden_velocity_alpha: float = 0.65
    hidden_acceleration_alpha: float = 0.35
    hidden_stationary_speed: float = 0.006
    hidden_motion_history: int = 8
    hidden_min_motion_history: int = 4
    hidden_stationary_displacement: float = 0.015
    hidden_moving_displacement: float = 0.035
    hidden_motion_consistency: float = 0.55
    hidden_stationary_lock_frames: int = 8
    hidden_max_motion_step_box_scale: float = 1.50
    default_behavior: str = "lying"
    smooth_boxes: bool = True
    refine_boxes: bool = True
    refine_max_gap_frames: int = 15
    refine_size_jump_threshold: float = 0.45
    max_box_scale_change_per_frame: float = 0.25
    max_box_scale_change_after_gap: float = 0.75
    high_conf_smooth_alpha: float = 0.75
    mid_conf_smooth_alpha: float = 0.55
    low_conf_smooth_alpha: float = 0.35

    max_frames: int | None = None
    draw_mask_outline: bool = True
    shade_outside_mask: bool = True
    visual_opacity: float = DEFAULT_VISUAL_OPACITY
    show: bool = False
    display_inline: bool = False


@dataclass(slots=True)
class Detection:
    """One YOLO/ByteTrack detection after ROI filtering."""

    box: np.ndarray
    score: float
    raw_id: int | None
    class_id: int | None
    hist: np.ndarray
    mask: np.ndarray | None = None


@dataclass(slots=True)
class OcclusionContext:
    """Ambiguous track/detection relationships in crowded overlap zones."""

    predicted_boxes: dict[int, np.ndarray]
    occluded_track_ids: set[int]
    detection_competitors: dict[int, set[int]]
    active_detection_owners: dict[int, set[int]]
    appearance_costs: dict[tuple[int, int], float]


@dataclass(slots=True)
class FixedTrack:
    """Stable ID state. Each video frame will emit one box per FixedTrack."""

    fixed_id: int
    last_box: np.ndarray
    reliable_box: np.ndarray | None = None
    hist_bank: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=80))
    raw_id_counts: Counter[int] = field(default_factory=Counter)
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    reliable_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, np.float32)
    )
    reliable_acceleration_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, np.float32)
    )
    last_mask: np.ndarray | None = None
    reliable_center_history: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=16)
    )
    reliable_velocity_history: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=16)
    )
    missed: int = 0
    occlusion_hold_frames: int = 0
    stationary_frames: int = 0
    motion_state: str = "unknown"
    hits: int = 0
    last_score: float = 0.0
    last_source: str = "placeholder"
    last_ambiguous: bool = False
    ever_detected: bool = False

    def mean_hist(self) -> np.ndarray | None:
        if not self.hist_bank:
            return None
        return np.mean(np.stack(tuple(self.hist_bank), axis=0), axis=0)

    def top_raw_id(self) -> int | None:
        if not self.raw_id_counts:
            return None
        return self.raw_id_counts.most_common(1)[0][0]

    def predicted_box(self, width: int, height: int) -> np.ndarray:
        damping = 0.85 ** min(self.missed, 12)
        if self.missed > 90:
            damping = 0.0
        dx, dy = self.velocity_xy * damping
        box = self.last_box.copy()
        box[[0, 2]] += dx
        box[[1, 3]] += dy
        return clip_box(box, width, height)

    def reliable_speed_norm(self, width: int, height: int) -> float:
        diag = math.sqrt(width * width + height * height)
        return float(np.linalg.norm(self.reliable_velocity_xy) / max(diag, 1e-6))

    def update_reliable_motion(
        self,
        new_box: np.ndarray,
        width: int,
        height: int,
        cfg: TrackingConfig,
    ) -> None:
        """Update hidden-only motion state from stable, non-ambiguous detections."""
        new_center = np.array(bbox_center(new_box), dtype=np.float32)
        if self.reliable_box is None or not self.reliable_center_history:
            self.reliable_box = new_box.copy()
            self.reliable_velocity_xy = np.zeros(2, np.float32)
            self.reliable_acceleration_xy = np.zeros(2, np.float32)
            self.reliable_center_history.append(new_center)
            self.stationary_frames = 0
            self.motion_state = "unknown"
            return

        previous_center = self.reliable_center_history[-1]
        instant_velocity = new_center - previous_center
        old_velocity = self.reliable_velocity_xy.copy()
        self.reliable_center_history.append(new_center)
        self.reliable_velocity_history.append(instant_velocity.astype(np.float32))

        while len(self.reliable_center_history) > cfg.hidden_motion_history:
            self.reliable_center_history.popleft()
        while len(self.reliable_velocity_history) > max(
            1,
            cfg.hidden_motion_history - 1,
        ):
            self.reliable_velocity_history.popleft()

        diag = math.sqrt(width * width + height * height)
        history_ready = (
            len(self.reliable_center_history) >= cfg.hidden_min_motion_history
        )
        centers = np.stack(tuple(self.reliable_center_history), axis=0)
        velocities = (
            np.stack(tuple(self.reliable_velocity_history), axis=0)
            if self.reliable_velocity_history
            else np.zeros((1, 2), dtype=np.float32)
        )
        total_displacement = float(
            np.linalg.norm(centers[-1] - centers[0]) / max(diag, 1e-6)
        )
        frame_span = max(1, len(centers) - 1)
        linear_speed = total_displacement / frame_span
        velocity_norms = np.linalg.norm(velocities, axis=1)
        median_speed = float(np.median(velocity_norms) / max(diag, 1e-6))
        path_length = float(np.sum(velocity_norms))
        net_displacement = float(np.linalg.norm(np.sum(velocities, axis=0)))
        consistency = net_displacement / max(path_length, 1e-6)
        recent_jitter = float(np.percentile(velocity_norms, 75) / max(diag, 1e-6))
        moving_by_displacement = total_displacement >= cfg.hidden_moving_displacement
        moving_by_velocity = (
            total_displacement > cfg.hidden_stationary_displacement
            and linear_speed >= cfg.hidden_stationary_speed * 0.65
        )

        if history_ready and (
            total_displacement <= cfg.hidden_stationary_displacement
            and median_speed <= cfg.hidden_stationary_speed
        ):
            self.stationary_frames += 1
            self.motion_state = "stationary"
            instant_velocity = np.zeros(2, np.float32)
        elif history_ready and (
            (moving_by_displacement or moving_by_velocity)
            and consistency >= cfg.hidden_motion_consistency
            and recent_jitter > cfg.hidden_stationary_speed * 0.65
        ):
            self.stationary_frames = 0
            self.motion_state = "moving"
            instant_velocity = np.mean(velocities, axis=0).astype(np.float32)
        else:
            if (
                history_ready
                and total_displacement > cfg.hidden_stationary_displacement
            ):
                self.stationary_frames = 0
            self.motion_state = "unknown"

        if self.motion_state != "moving":
            instant_velocity = np.zeros(2, np.float32)

        velocity_alpha = cfg.hidden_velocity_alpha
        acceleration_alpha = cfg.hidden_acceleration_alpha
        self.reliable_velocity_xy = (
            velocity_alpha * instant_velocity
            + (1.0 - velocity_alpha) * self.reliable_velocity_xy
        ).astype(np.float32)
        instant_acceleration = instant_velocity - old_velocity
        self.reliable_acceleration_xy = (
            acceleration_alpha * instant_acceleration
            + (1.0 - acceleration_alpha) * self.reliable_acceleration_xy
        ).astype(np.float32)
        if self.stationary_frames >= cfg.hidden_stationary_lock_frames:
            self.motion_state = "stationary"
            self.reliable_velocity_xy *= 0.0
            self.reliable_acceleration_xy *= 0.0
        self.reliable_box = new_box.copy()

    def hidden_motion_box(
        self,
        width: int,
        height: int,
        cfg: TrackingConfig,
    ) -> np.ndarray:
        """Predict only when the animal is hidden/fully occluded."""
        if not cfg.hidden_motion_model:
            reference = (
                self.reliable_box if self.reliable_box is not None else self.last_box
            )
            return clip_box(reference.copy(), width, height)

        if self.motion_state != "moving":
            reference = (
                self.reliable_box if self.reliable_box is not None else self.last_box
            )
            return clip_box(reference.copy(), width, height)

        speed = self.reliable_speed_norm(width, height)
        if speed <= cfg.hidden_stationary_speed:
            reference = (
                self.reliable_box if self.reliable_box is not None else self.last_box
            )
            return clip_box(reference.copy(), width, height)

        if self.occlusion_hold_frames == 0 and self.reliable_box is not None:
            reference = self.reliable_box.copy()
        else:
            reference = self.last_box.copy()
        displacement = self.reliable_velocity_xy + 0.5 * self.reliable_acceleration_xy
        displacement *= 0.90 ** min(self.missed + self.occlusion_hold_frames, 12)

        box_w, box_h = bbox_size(reference)
        max_step = cfg.hidden_max_motion_step_box_scale * math.hypot(box_w, box_h)
        step_norm = float(np.linalg.norm(displacement))
        if max_step > 0.0 and step_norm > max_step:
            displacement *= max_step / max(step_norm, 1e-6)

        predicted = reference.copy()
        predicted[[0, 2]] += displacement[0]
        predicted[[1, 3]] += displacement[1]
        return clip_box(predicted, width, height)

    def update_detected(
        self,
        det: Detection,
        width: int,
        height: int,
        cfg: TrackingConfig,
        learn_identity: bool = True,
        ambiguous: bool = False,
    ) -> None:
        previous_box = self.last_box
        if (
            self.ever_detected
            and self.missed > 0
            and detection_needs_motion_gate(det, cfg)
            and self.reliable_box is not None
        ):
            previous_box = self.reliable_box

        previous_center = bbox_center(previous_box)
        new_box = clip_box(det.box, width, height)
        if self.ever_detected and cfg.smooth_boxes:
            new_box = smooth_detected_box(
                previous_box,
                new_box,
                det.score,
                self.missed,
                width,
                height,
                cfg,
            )
        new_center = bbox_center(new_box)
        self.velocity_xy = np.array(
            [new_center[0] - previous_center[0], new_center[1] - previous_center[1]],
            dtype=np.float32,
        )
        self.last_box = new_box
        self.last_score = float(det.score)
        self.last_source = "detected"
        self.last_ambiguous = bool(ambiguous)
        self.last_mask = det.mask.copy() if det.mask is not None else None
        self.missed = 0
        self.hits += 1
        self.ever_detected = True
        if learn_identity:
            self.hist_bank.append(det.hist)
        if learn_identity and det.raw_id is not None:
            self.raw_id_counts.update([int(det.raw_id)])
        if learn_identity and not ambiguous and det.score >= cfg.track_high_conf:
            self.update_reliable_motion(new_box, width, height, cfg)
            self.occlusion_hold_frames = 0

    def update_predicted(
        self,
        box: np.ndarray,
        width: int,
        height: int,
        ambiguous: bool = False,
        hold: bool = False,
    ) -> None:
        previous_center = bbox_center(self.last_box)
        new_box = clip_box(box, width, height)
        new_center = bbox_center(new_box)
        self.velocity_xy = 0.5 * self.velocity_xy + 0.5 * np.array(
            [new_center[0] - previous_center[0], new_center[1] - previous_center[1]],
            dtype=np.float32,
        )
        self.last_box = new_box
        self.last_score = max(0.05, self.last_score * 0.92)
        self.last_source = "occlusion_hold" if hold else "predicted"
        self.last_ambiguous = bool(ambiguous)
        self.missed += 1
        if hold:
            self.occlusion_hold_frames += 1
        else:
            self.occlusion_hold_frames = 0


@dataclass(slots=True)
class TrackingSummary:
    """Paths and counters returned by run_tracking."""

    output_video: Path
    annotations_json: Path
    coco_annotations_json: Path
    clean_coco_annotations_json: Path
    cvat_video_xml: Path
    labels_json: Path
    quality_report_json: Path
    quality_report_csv: Path
    frames_read: int
    frames_written: int
    shape_count: int
    hidden_shape_count: int
    review_shape_count: int
    start_frame: int
    source_fps: float
    output_fps: float


# %%
def validate_config(cfg: TrackingConfig) -> None:
    if cfg.conf is not None:
        cfg.review_conf = cfg.conf
    if cfg.start_frame < 0:
        raise ValueError("start_frame must be >= 0.")
    if cfg.expected_pigs != len(ID_VALUES):
        raise ValueError("The CVAT label schema is fixed to exactly 8 pig IDs.")
    if cfg.default_behavior not in BEHAVIOR_VALUES:
        raise ValueError(f"default_behavior must be one of: {BEHAVIOR_VALUES}")
    if cfg.roi_mode not in {"center", "cover"}:
        raise ValueError("roi_mode must be either 'center' or 'cover'.")
    confidence_values = {
        "det_conf": cfg.det_conf,
        "track_high_conf": cfg.track_high_conf,
        "review_conf": cfg.review_conf,
    }
    for name, value in confidence_values.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    gate_confidence_values = {
        "initial_track_conf": cfg.initial_track_conf,
        "motion_gate_confidence": cfg.motion_gate_confidence,
    }
    for name, value in gate_confidence_values.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.det_conf > cfg.track_high_conf:
        raise ValueError("det_conf should be <= track_high_conf.")
    if cfg.track_high_conf > cfg.review_conf:
        raise ValueError("track_high_conf should be <= review_conf.")
    if cfg.initial_track_conf < cfg.det_conf:
        raise ValueError("initial_track_conf should be >= det_conf.")
    if cfg.motion_gate_confidence < cfg.det_conf:
        raise ValueError("motion_gate_confidence should be >= det_conf.")
    if not 0.0 < cfg.adaptive_conf_step <= 0.50:
        raise ValueError("adaptive_conf_step must be between 0 and 0.50.")
    if not 0.0 < cfg.iou < 1.0:
        raise ValueError("iou must be between 0 and 1.")
    if not 0.0 <= cfg.visual_opacity <= 1.0:
        raise ValueError("visual_opacity must be between 0 and 1.")
    if cfg.hidden_missed_frames < 1:
        raise ValueError("hidden_missed_frames must be >= 1.")
    if not 0.0 <= cfg.hidden_score_threshold <= 1.0:
        raise ValueError("hidden_score_threshold must be between 0 and 1.")
    if cfg.mask_iou_max_missed < 0:
        raise ValueError("mask_iou_max_missed must be >= 0.")
    if cfg.mask_iou_min_area < 1:
        raise ValueError("mask_iou_min_area must be >= 1.")
    cost_thresholds = {
        "match_cost_threshold": cfg.match_cost_threshold,
        "unseen_track_cost_threshold": cfg.unseen_track_cost_threshold,
        "lost_track_cost_threshold": cfg.lost_track_cost_threshold,
    }
    for name, value in cost_thresholds.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")
    if not 0.0 <= cfg.lost_track_reid_appearance_threshold <= 1.0:
        raise ValueError(
            "lost_track_reid_appearance_threshold must be between 0 and 1."
        )
    if not 0.0 <= cfg.low_conf_max_center_jump <= 1.0:
        raise ValueError("low_conf_max_center_jump must be between 0 and 1.")
    if not 0.0 <= cfg.low_conf_min_iou <= 1.0:
        raise ValueError("low_conf_min_iou must be between 0 and 1.")
    if cfg.low_conf_max_box_jump_scale < 0.0:
        raise ValueError("low_conf_max_box_jump_scale must be >= 0.")
    occlusion_values = {
        "occlusion_track_iom_threshold": cfg.occlusion_track_iom_threshold,
        "occlusion_detection_iom_threshold": cfg.occlusion_detection_iom_threshold,
        "occlusion_stationary_speed": cfg.occlusion_stationary_speed,
        "occlusion_stationary_max_center_jump": (
            cfg.occlusion_stationary_max_center_jump
        ),
        "occlusion_switch_penalty": cfg.occlusion_switch_penalty,
        "occlusion_competitor_margin": cfg.occlusion_competitor_margin,
        "occlusion_appearance_penalty": cfg.occlusion_appearance_penalty,
        "occlusion_appearance_margin": cfg.occlusion_appearance_margin,
    }
    for name, value in occlusion_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.occlusion_hold_max_frames < 0:
        raise ValueError("occlusion_hold_max_frames must be >= 0.")
    if cfg.occlusion_hold_hidden_frames < 1:
        raise ValueError("occlusion_hold_hidden_frames must be >= 1.")
    identity_swap_values = {
        "identity_swap_min_gain": cfg.identity_swap_min_gain,
        "identity_swap_iom_threshold": cfg.identity_swap_iom_threshold,
    }
    for name, value in identity_swap_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    hidden_motion_values = {
        "hidden_velocity_alpha": cfg.hidden_velocity_alpha,
        "hidden_acceleration_alpha": cfg.hidden_acceleration_alpha,
        "hidden_stationary_speed": cfg.hidden_stationary_speed,
        "hidden_stationary_displacement": cfg.hidden_stationary_displacement,
        "hidden_moving_displacement": cfg.hidden_moving_displacement,
        "hidden_motion_consistency": cfg.hidden_motion_consistency,
    }
    for name, value in hidden_motion_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.hidden_motion_history < 2:
        raise ValueError("hidden_motion_history must be >= 2.")
    if cfg.hidden_min_motion_history < 2:
        raise ValueError("hidden_min_motion_history must be >= 2.")
    if cfg.hidden_min_motion_history > cfg.hidden_motion_history:
        raise ValueError(
            "hidden_min_motion_history must be <= hidden_motion_history."
        )
    if cfg.hidden_stationary_lock_frames < 1:
        raise ValueError("hidden_stationary_lock_frames must be >= 1.")
    if cfg.hidden_max_motion_step_box_scale < 0.0:
        raise ValueError("hidden_max_motion_step_box_scale must be >= 0.")
    scale_values = {
        "max_box_scale_change_per_frame": cfg.max_box_scale_change_per_frame,
        "max_box_scale_change_after_gap": cfg.max_box_scale_change_after_gap,
    }
    for name, value in scale_values.items():
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"{name} must be between 0 and 2.")
    if cfg.refine_max_gap_frames < 1:
        raise ValueError("refine_max_gap_frames must be >= 1.")
    if not 0.0 <= cfg.refine_size_jump_threshold <= 2.0:
        raise ValueError("refine_size_jump_threshold must be between 0 and 2.")
    alpha_values = {
        "high_conf_smooth_alpha": cfg.high_conf_smooth_alpha,
        "mid_conf_smooth_alpha": cfg.mid_conf_smooth_alpha,
        "low_conf_smooth_alpha": cfg.low_conf_smooth_alpha,
    }
    for name, value in alpha_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")
    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"YOLOv8 weights not found: {cfg.weights_path}")
    if cfg.use_mask and cfg.mask_path is not None and not cfg.mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {cfg.mask_path}")


def resolve_output_paths(
    cfg: TrackingConfig,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    video_stem = cfg.video_path.stem
    run_output_dir = cfg.output_dir / video_stem
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video = cfg.output_video or (
        run_output_dir / f"{video_stem}_tracked_pigs_with_ids.mp4"
    )
    annotations_json = cfg.annotations_json or (
        run_output_dir / f"{video_stem}_annotations_cvat_shapes.json"
    )
    coco_annotations_json = cfg.coco_annotations_json or (
        run_output_dir / f"{video_stem}_annotations_coco.json"
    )
    clean_coco_annotations_json = cfg.clean_coco_annotations_json or (
        run_output_dir / f"{video_stem}_annotations_coco_clean_train.json"
    )
    cvat_video_xml = cfg.cvat_video_xml or (
        run_output_dir / f"{video_stem}_annotations_cvat_video_1_1.xml"
    )
    labels_json = cfg.labels_json or run_output_dir / f"{video_stem}_labels.json"
    tracker_yaml = cfg.tracker_yaml or (
        run_output_dir / f"{video_stem}_bytetrack_pig_8.yaml"
    )
    quality_report_json = cfg.quality_report_json or (
        run_output_dir / f"{video_stem}_tracking_quality_report.json"
    )
    quality_report_csv = cfg.quality_report_csv or (
        run_output_dir / f"{video_stem}_tracking_quality_report.csv"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    annotations_json.parent.mkdir(parents=True, exist_ok=True)
    coco_annotations_json.parent.mkdir(parents=True, exist_ok=True)
    clean_coco_annotations_json.parent.mkdir(parents=True, exist_ok=True)
    cvat_video_xml.parent.mkdir(parents=True, exist_ok=True)
    labels_json.parent.mkdir(parents=True, exist_ok=True)
    tracker_yaml.parent.mkdir(parents=True, exist_ok=True)
    quality_report_json.parent.mkdir(parents=True, exist_ok=True)
    quality_report_csv.parent.mkdir(parents=True, exist_ok=True)
    return (
        output_video,
        annotations_json,
        coco_annotations_json,
        clean_coco_annotations_json,
        cvat_video_xml,
        labels_json,
        tracker_yaml,
        quality_report_json,
        quality_report_csv,
    )


def write_tracker_yaml(path: Path, cfg: TrackingConfig) -> None:
    """Write a ByteTrack config tuned for crowded 30 FPS pig videos."""
    track_low_thresh = min(cfg.det_conf, cfg.track_high_conf)
    path.write_text(
        "\n".join(
            [
                "tracker_type: bytetrack",
                f"track_high_thresh: {cfg.track_high_conf:.2f}",
                f"track_low_thresh: {track_low_thresh:.2f}",
                f"new_track_thresh: {cfg.track_high_conf:.2f}",
                f"track_thresh: {cfg.track_high_conf:.2f}",
                f"match_thresh: {cfg.iou:.2f}",
                "track_buffer: 90",
                "min_box_area: 10",
                "mot20: false",
                "fuse_score: true",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.25",
                "max_age: 90",
                "n_init: 3",
                "with_reid: true",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def _result_masks(
    result: Any,
    width: int,
    height: int,
) -> list[np.ndarray | None]:
    masks = getattr(result, "masks", None)
    data = _to_numpy(getattr(masks, "data", None)) if masks is not None else None
    if data is None or len(data) == 0:
        return []

    import cv2

    out: list[np.ndarray | None] = []
    for mask_values in data:
        mask = np.asarray(mask_values)
        if mask.ndim != 2:
            out.append(None)
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        out.append(mask > 0.5)
    return out


# %%
def load_mask(
    mask_path: Path | None,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray | None:
    if not cfg.use_mask or mask_path is None:
        return None

    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if cfg.roi_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.roi_dilate_px, cfg.roi_dilate_px),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_mask_to_frame(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame

    import cv2

    return cv2.bitwise_and(frame, frame, mask=mask)


def shade_outside_roi(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame.copy()
    shaded = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    out = frame.copy()
    out[mask == 0] = shaded[mask == 0]
    return out


def roi_keep(mask: np.ndarray | None, box: np.ndarray, cfg: TrackingConfig) -> bool:
    if mask is None:
        return True

    height, width = mask.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return False

    if cfg.roi_mode == "center":
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        return bool(mask[cy, cx] == 255)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    cover = np.count_nonzero(roi == 255) / float(roi.size)
    return cover >= cfg.roi_min_cover


def mask_anchor_boxes(
    mask: np.ndarray | None,
    width: int,
    height: int,
    count: int,
    median_box: np.ndarray | None,
) -> list[np.ndarray]:
    """Create hidden fallback boxes so every frame can contain 8 shapes."""
    if mask is not None and np.count_nonzero(mask) > 0:
        ys, xs = np.where(mask > 0)
        rx1, rx2 = float(xs.min()), float(xs.max())
        ry1, ry2 = float(ys.min()), float(ys.max())
    else:
        rx1, ry1, rx2, ry2 = 0.0, 0.0, float(width - 1), float(height - 1)

    roi_w = max(1.0, rx2 - rx1)
    roi_h = max(1.0, ry2 - ry1)
    if median_box is not None:
        bw = max(24.0, float(median_box[2] - median_box[0]))
        bh = max(24.0, float(median_box[3] - median_box[1]))
    else:
        bw = max(24.0, roi_w * 0.18)
        bh = max(24.0, roi_h * 0.22)

    cols = 4
    rows = int(math.ceil(count / cols))
    boxes: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            if len(boxes) == count:
                break
            cx = rx1 + (col + 0.5) * roi_w / cols
            cy = ry1 + (row + 0.5) * roi_h / rows
            box = np.array(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                dtype=np.float32,
            )
            boxes.append(clip_box(box, width, height))
    return boxes


# %%
def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float32).copy()
    out[0] = max(0.0, min(float(width - 1), float(out[0])))
    out[1] = max(0.0, min(float(height - 1), float(out[1])))
    out[2] = max(out[0] + 1.0, min(float(width), float(out[2])))
    out[3] = max(out[1] + 1.0, min(float(height), float(out[3])))
    return out


def bbox_area(box: np.ndarray) -> float:
    return max(1.0, float(box[2] - box[0])) * max(1.0, float(box[3] - box[1]))


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def bbox_size(box: np.ndarray) -> tuple[float, float]:
    return max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1]))


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / max(union, 1e-6)


def mask_area(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    first_bool = first.astype(bool, copy=False)
    second_bool = second.astype(bool, copy=False)
    inter = int(np.logical_and(first_bool, second_bool).sum())
    union = int(np.logical_or(first_bool, second_bool).sum())
    if union <= 0:
        return None
    return float(inter / union)


def shift_mask(
    mask: np.ndarray | None,
    dx: float,
    dy: float,
) -> np.ndarray | None:
    """Translate a binary mask by a small integer offset without wraparound."""
    if mask is None:
        return None
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    if shift_x == 0 and shift_y == 0:
        return mask

    height, width = mask.shape[:2]
    shifted = np.zeros_like(mask, dtype=bool)
    src_x1 = max(0, -shift_x)
    src_x2 = min(width, width - shift_x)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(width, width + shift_x)
    src_y1 = max(0, -shift_y)
    src_y2 = min(height, height - shift_y)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(height, height + shift_y)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return shifted
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def track_mask_for_box(
    track: FixedTrack,
    predicted_box: np.ndarray,
    cfg: TrackingConfig,
) -> np.ndarray | None:
    if (
        not cfg.use_mask_iou
        or track.last_mask is None
        or track.missed > cfg.mask_iou_max_missed
        or mask_area(track.last_mask) < cfg.mask_iou_min_area
    ):
        return None
    last_cx, last_cy = bbox_center(track.last_box)
    pred_cx, pred_cy = bbox_center(predicted_box)
    return shift_mask(track.last_mask, pred_cx - last_cx, pred_cy - last_cy)


def detection_overlap_score(first: Detection, second: Detection, cfg: TrackingConfig) -> float:
    if (
        cfg.use_mask_iou
        and mask_area(first.mask) >= cfg.mask_iou_min_area
        and mask_area(second.mask) >= cfg.mask_iou_min_area
    ):
        score = mask_iou(first.mask, second.mask)
        if score is not None:
            return score
    return bbox_iou(first.box, second.box)


def track_detection_overlap_score(
    track: FixedTrack,
    predicted_box: np.ndarray,
    det: Detection,
    cfg: TrackingConfig,
) -> float:
    if cfg.use_mask_iou and mask_area(det.mask) >= cfg.mask_iou_min_area:
        score = mask_iou(track_mask_for_box(track, predicted_box, cfg), det.mask)
        if score is not None:
            return score
    return bbox_iou(predicted_box, det.box)


def bbox_intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iom(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over the smaller box area, useful for occlusion detection."""
    inter = bbox_intersection_area(first, second)
    return inter / max(min(bbox_area(first), bbox_area(second)), 1e-6)


def center_distance_norm(
    first: np.ndarray,
    second: np.ndarray,
    width: int,
    height: int,
) -> float:
    cx1, cy1 = bbox_center(first)
    cx2, cy2 = bbox_center(second)
    diag = math.sqrt(width * width + height * height)
    return math.dist((cx1, cy1), (cx2, cy2)) / max(diag, 1e-6)


def area_log_ratio(first: np.ndarray, second: np.ndarray) -> float:
    return abs(math.log((bbox_area(second) + 1e-6) / (bbox_area(first) + 1e-6)))


def smooth_alpha_for_score(score: float, cfg: TrackingConfig) -> float:
    if score >= cfg.review_conf:
        return cfg.high_conf_smooth_alpha
    if score >= cfg.track_high_conf:
        return cfg.mid_conf_smooth_alpha
    return cfg.low_conf_smooth_alpha


def smooth_detected_box(
    previous_box: np.ndarray,
    detected_box: np.ndarray,
    score: float,
    missed_frames: int,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    """Limit sudden box-size jumps while keeping motion responsive."""
    previous_box = clip_box(previous_box, width, height)
    detected_box = clip_box(detected_box, width, height)
    prev_cx, prev_cy = bbox_center(previous_box)
    det_cx, det_cy = bbox_center(detected_box)
    prev_w, prev_h = bbox_size(previous_box)
    det_w, det_h = bbox_size(detected_box)

    max_scale_change = (
        cfg.max_box_scale_change_after_gap
        if missed_frames > 0
        else cfg.max_box_scale_change_per_frame
    )
    min_scale = max(0.05, 1.0 - max_scale_change)
    max_scale = 1.0 + max_scale_change
    limited_w = float(np.clip(det_w, prev_w * min_scale, prev_w * max_scale))
    limited_h = float(np.clip(det_h, prev_h * min_scale, prev_h * max_scale))

    alpha = smooth_alpha_for_score(score, cfg)
    center_alpha = max(alpha, 0.80)
    cx = center_alpha * det_cx + (1.0 - center_alpha) * prev_cx
    cy = center_alpha * det_cy + (1.0 - center_alpha) * prev_cy
    smooth_w = alpha * limited_w + (1.0 - alpha) * prev_w
    smooth_h = alpha * limited_h + (1.0 - alpha) * prev_h

    smoothed = np.array(
        [
            cx - smooth_w / 2.0,
            cy - smooth_h / 2.0,
            cx + smooth_w / 2.0,
            cy + smooth_h / 2.0,
        ],
        dtype=np.float32,
    )
    return clip_box(smoothed, width, height)


def extract_hist_hsv(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [16, 16, 4],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist.flatten()


def hist_distance(first: np.ndarray | None, second: np.ndarray) -> float:
    if first is None:
        return 0.50
    return float(np.clip(1.0 - np.sum(np.sqrt(first * second)), 0.0, 1.0))


def lk_predict_box(
    prev_frame: np.ndarray | None,
    frame: np.ndarray,
    last_box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Predict a missing box with Lucas-Kanade optical flow."""
    if prev_frame is None:
        return None

    import cv2

    x1, y1, x2, y2 = clip_box(last_box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = prev_gray[y1:y2, x1:x2]
    if roi.size < 64:
        return None

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=60,
        qualityLevel=0.01,
        minDistance=4,
    )
    if points is None:
        return None
    points = points.reshape(-1, 1, 2)
    points[:, :, 0] += x1
    points[:, :, 1] += y1

    next_points, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    if next_points is None or status is None:
        return None
    good = status.reshape(-1) == 1
    if not np.any(good):
        return None

    delta = (next_points[good] - points[good]).reshape(-1, 2)
    dx = float(np.median(delta[:, 0]))
    dy = float(np.median(delta[:, 1]))
    predicted = last_box.copy()
    predicted[[0, 2]] += dx
    predicted[[1, 3]] += dy
    return clip_box(predicted, width, height)


# %%
def parse_detections(
    result: Any,
    frame: np.ndarray,
    mask: np.ndarray | None,
    cfg: TrackingConfig,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = frame.shape[:2]
    names = _names_dict(getattr(result, "names", {}))
    xyxy = _to_numpy(boxes.xyxy)
    conf = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    raw_ids = _to_numpy(getattr(boxes, "id", None))
    masks = _result_masks(result, width, height)
    if xyxy is None or conf is None:
        return []

    detections: list[Detection] = []
    for idx, box_values in enumerate(xyxy):
        class_id = int(classes[idx]) if classes is not None else None
        if cfg.class_id is not None and class_id != cfg.class_id:
            continue
        if cfg.allowed_class_name is not None and class_id is not None:
            class_name = names.get(class_id, "").lower()
            if class_name != cfg.allowed_class_name.lower():
                continue

        box = clip_box(np.asarray(box_values, dtype=np.float32), width, height)
        if not roi_keep(mask, box, cfg):
            continue

        raw_id = None
        if raw_ids is not None and idx < len(raw_ids):
            raw_id = int(raw_ids[idx])
        detections.append(
            Detection(
                box=box,
                score=float(conf[idx]),
                raw_id=raw_id,
                class_id=class_id,
                hist=extract_hist_hsv(frame, box),
                mask=masks[idx] if idx < len(masks) else None,
            )
        )

    detections.sort(key=lambda item: item.score, reverse=True)
    detections = suppress_duplicate_detections(detections, cfg)
    return detections[: max(cfg.expected_pigs * 3, cfg.expected_pigs)]


def suppress_duplicate_detections(
    detections: list[Detection],
    cfg: TrackingConfig,
) -> list[Detection]:
    kept: list[Detection] = []
    for det in detections:
        if all(
            detection_overlap_score(det, other, cfg) < cfg.duplicate_iou_threshold
            for other in kept
        ):
            kept.append(det)
    return kept


def confidence_ladder(cfg: TrackingConfig) -> list[float]:
    """Return descending thresholds from review_conf to det_conf."""
    thresholds: list[float] = []
    current = cfg.review_conf
    while current > cfg.det_conf:
        thresholds.append(round(current, 4))
        current -= cfg.adaptive_conf_step
    thresholds.append(round(cfg.det_conf, 4))

    unique_thresholds: list[float] = []
    seen: set[float] = set()
    for threshold in thresholds:
        clipped = float(np.clip(threshold, cfg.det_conf, cfg.review_conf))
        if clipped not in seen:
            unique_thresholds.append(clipped)
            seen.add(clipped)
    return unique_thresholds


def adaptive_confidence_filter(
    detections: list[Detection],
    cfg: TrackingConfig,
) -> list[Detection]:
    """Keep the highest confidence threshold that still gives enough candidates."""
    if not detections:
        return []

    max_candidates = max(cfg.expected_pigs * 3, cfg.expected_pigs)
    for threshold in confidence_ladder(cfg):
        selected = [det for det in detections if det.score >= threshold]
        if len(selected) >= cfg.expected_pigs:
            return selected[:max_candidates]
    return detections[:max_candidates]


def spatial_sort_detections(detections: list[Detection]) -> list[Detection]:
    return sorted(
        detections,
        key=lambda det: (bbox_center(det.box)[1], bbox_center(det.box)[0]),
    )


def initialize_tracks(
    detections: list[Detection],
    mask: np.ndarray | None,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> dict[int, FixedTrack]:
    init_detections = [
        det for det in detections if det.score >= cfg.initial_track_conf
    ]
    selected = init_detections[: cfg.expected_pigs]
    median_box = None
    if selected:
        widths = [det.box[2] - det.box[0] for det in selected]
        heights = [det.box[3] - det.box[1] for det in selected]
        median_w = float(np.median(widths))
        median_h = float(np.median(heights))
        median_box = np.array([0.0, 0.0, median_w, median_h], dtype=np.float32)

    anchors = mask_anchor_boxes(mask, width, height, cfg.expected_pigs, median_box)
    anchor_detection_pairs = sorted(
        (
            (
                center_distance_norm(anchor, det.box, width, height),
                anchor_idx,
                det_idx,
            )
            for anchor_idx, anchor in enumerate(anchors)
            for det_idx, det in enumerate(selected)
        ),
        key=lambda item: item[0],
    )
    used_anchor_idx: set[int] = set()
    used_det_idx: set[int] = set()
    tracks: dict[int, FixedTrack] = {}

    for _cost, anchor_idx, det_idx in anchor_detection_pairs:
        if anchor_idx in used_anchor_idx or det_idx in used_det_idx:
            continue
        fixed_id = anchor_idx + 1
        det = selected[det_idx]
        used_anchor_idx.add(anchor_idx)
        used_det_idx.add(det_idx)
        track = FixedTrack(fixed_id=fixed_id, last_box=det.box.copy())
        track.update_detected(det, width, height, cfg)
        tracks[fixed_id] = track

    for fixed_id in range(1, cfg.expected_pigs + 1):
        if fixed_id not in tracks:
            tracks[fixed_id] = FixedTrack(
                fixed_id=fixed_id,
                last_box=anchors[fixed_id - 1].copy(),
            )

    return tracks


# %%
def detection_needs_motion_gate(det: Detection, cfg: TrackingConfig) -> bool:
    return cfg.low_conf_motion_gate and det.score < cfg.motion_gate_confidence


def track_is_visible_for_association(track: FixedTrack) -> bool:
    return (
        track.ever_detected
        and track.missed == 0
        and track.last_source == "detected"
        and not track.last_ambiguous
    )


def track_is_lost_for_association(track: FixedTrack) -> bool:
    return track.ever_detected and not track_is_visible_for_association(track)


def association_reference_box(
    track: FixedTrack,
    det: Detection,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    if detection_needs_motion_gate(det, cfg) and track.ever_detected:
        reference = (
            track.reliable_box if track.reliable_box is not None else track.last_box
        )
        return clip_box(reference.copy(), width, height)
    return track.predicted_box(width, height)


def low_conf_detection_is_plausible(
    track: FixedTrack,
    det: Detection,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    """Reject low-confidence detections that jump far from a known track."""
    if not detection_needs_motion_gate(det, cfg):
        return True
    if not track.ever_detected:
        return det.score >= cfg.initial_track_conf

    reference = association_reference_box(track, det, width, height, cfg)
    if track_is_lost_for_association(track):
        top_raw_id = track.top_raw_id()
        if (
            det.raw_id is not None
            and top_raw_id is not None
            and det.raw_id == top_raw_id
        ):
            return True
        if hist_distance(track.mean_hist(), det.hist) <= (
            cfg.lost_track_reid_appearance_threshold
        ):
            return True

    iou_score = track_detection_overlap_score(track, reference, det, cfg)
    if iou_score >= cfg.low_conf_min_iou:
        return True

    center_norm = center_distance_norm(reference, det.box, width, height)
    missed_growth = 0.008 if track_is_lost_for_association(track) else 0.004
    allowed_norm = cfg.low_conf_max_center_jump + min(track.missed, 30) * missed_growth
    if center_norm <= allowed_norm:
        return True

    pred_cx, pred_cy = bbox_center(reference)
    det_cx, det_cy = bbox_center(det.box)
    pred_w, pred_h = bbox_size(reference)
    allowed_px = cfg.low_conf_max_box_jump_scale * math.hypot(pred_w, pred_h)
    center_px = math.dist((pred_cx, pred_cy), (det_cx, det_cy))
    return center_px <= allowed_px


def track_speed_norm(track: FixedTrack, width: int, height: int) -> float:
    diag = math.sqrt(width * width + height * height)
    return float(np.linalg.norm(track.velocity_xy) / max(diag, 1e-6))


def track_is_stationary_locked(
    track: FixedTrack,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.occlusion_stationary_lock:
        return False
    if track.motion_state == "moving":
        return False
    return (
        track.stationary_frames >= cfg.hidden_stationary_lock_frames
        and track.reliable_speed_norm(width, height) <= cfg.occlusion_stationary_speed
    )


def build_occlusion_context(
    ordered_tracks: list[FixedTrack],
    detections: list[Detection],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> OcclusionContext:
    predicted_boxes = {
        track.fixed_id: track.predicted_box(width, height)
        for track in ordered_tracks
    }
    appearance_costs = {
        (det_idx, track.fixed_id): hist_distance(track.mean_hist(), det.hist)
        for det_idx, det in enumerate(detections)
        for track in ordered_tracks
    }
    if not cfg.occlusion_aware_matching:
        return OcclusionContext(predicted_boxes, set(), {}, {}, appearance_costs)

    occluded_track_ids: set[int] = set()
    for idx, first in enumerate(ordered_tracks):
        first_box = predicted_boxes[first.fixed_id]
        for second in ordered_tracks[idx + 1 :]:
            second_box = predicted_boxes[second.fixed_id]
            if bbox_iom(first_box, second_box) >= cfg.occlusion_track_iom_threshold:
                occluded_track_ids.update({first.fixed_id, second.fixed_id})

    detection_competitors: dict[int, set[int]] = {}
    for det_idx, det in enumerate(detections):
        competitors = {
            track.fixed_id
            for track in ordered_tracks
            if bbox_iom(predicted_boxes[track.fixed_id], det.box)
            >= cfg.occlusion_detection_iom_threshold
        }
        if len(competitors) > 1:
            detection_competitors[det_idx] = competitors
            occluded_track_ids.update(competitors)

    active_detection_owners: dict[int, set[int]] = {}
    for det_idx, det in enumerate(detections):
        owners = {
            track.fixed_id
            for track in ordered_tracks
            if track.ever_detected
            and track.missed == 0
            and track.last_source == "detected"
            and not track.last_ambiguous
            and (
                bbox_iom(predicted_boxes[track.fixed_id], det.box)
                >= cfg.occlusion_detection_iom_threshold
                or center_distance_norm(
                    predicted_boxes[track.fixed_id],
                    det.box,
                    width,
                    height,
                )
                <= cfg.low_conf_max_center_jump
            )
        }
        if owners:
            active_detection_owners[det_idx] = owners

    return OcclusionContext(
        predicted_boxes=predicted_boxes,
        occluded_track_ids=occluded_track_ids,
        detection_competitors=detection_competitors,
        active_detection_owners=active_detection_owners,
        appearance_costs=appearance_costs,
    )


def assignment_is_occlusion_ambiguous(
    track: FixedTrack,
    det_index: int,
    context: OcclusionContext,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.occlusion_aware_matching:
        return False
    competitors = context.detection_competitors.get(det_index, set())
    return track.fixed_id in context.occluded_track_ids or len(competitors) > 1


def should_hold_occluded_track_box(
    track: FixedTrack,
    detections: list[Detection],
    context: OcclusionContext,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.hold_occluded_box or not track.ever_detected:
        return False
    if track.occlusion_hold_frames >= cfg.occlusion_hold_max_frames:
        return False
    reference = track.reliable_box if track.reliable_box is not None else track.last_box
    if track.fixed_id in context.occluded_track_ids or track.last_ambiguous:
        return True
    if 0 < track.occlusion_hold_frames < cfg.occlusion_hold_max_frames:
        return True
    return any(
        bbox_iom(reference, det.box) >= cfg.occlusion_detection_iom_threshold
        for det in detections
    )


def occlusion_assignment_penalty(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> float:
    if not cfg.occlusion_aware_matching or not track.ever_detected:
        return 0.0

    competitors = context.detection_competitors.get(det_index, set())
    in_occlusion = track.fixed_id in context.occluded_track_ids or len(competitors) > 1
    if not in_occlusion:
        return 0.0

    predicted = context.predicted_boxes.get(
        track.fixed_id,
        track.predicted_box(width, height),
    )
    center_cost = center_distance_norm(predicted, det.box, width, height)
    stationary_allowed = (
        cfg.occlusion_stationary_max_center_jump + min(track.missed, 15) * 0.003
    )
    if track_is_stationary_locked(track, width, height, cfg) and (
        center_cost > stationary_allowed
    ):
        return 1_000_000.0

    penalty = 0.0
    if competitors:
        own_overlap = bbox_iom(predicted, det.box)
        other_overlaps = [
            bbox_iom(context.predicted_boxes[other_id], det.box)
            for other_id in competitors
            if other_id != track.fixed_id and other_id in context.predicted_boxes
        ]
        best_other_overlap = max(other_overlaps, default=0.0)
        if track.fixed_id not in competitors:
            penalty += cfg.occlusion_switch_penalty
        elif best_other_overlap > own_overlap + cfg.occlusion_competitor_margin:
            penalty += cfg.occlusion_switch_penalty
        elif competitors:
            own_app = context.appearance_costs.get((det_index, track.fixed_id), 0.5)
            best_other_app = min(
                (
                    context.appearance_costs.get((det_index, other_id), 0.5)
                    for other_id in competitors
                    if other_id != track.fixed_id
                ),
                default=0.5,
            )
            if own_app > best_other_app + cfg.occlusion_appearance_margin:
                penalty += cfg.occlusion_appearance_penalty

    top_raw_id = track.top_raw_id()
    if det.raw_id is not None and top_raw_id is not None and det.raw_id != top_raw_id:
        penalty += 0.15

    return float(penalty)


def detection_is_reserved_for_active_track(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    """Keep reacquiring hidden tracks from stealing an active track's detection."""
    if track.missed == 0 and track.last_source == "detected":
        return False

    owners = context.active_detection_owners.get(det_index, set()) - {track.fixed_id}
    if not owners:
        return False

    top_raw_id = track.top_raw_id()
    if det.raw_id is not None and top_raw_id is not None and det.raw_id == top_raw_id:
        return False

    predicted = context.predicted_boxes.get(
        track.fixed_id,
        track.predicted_box(width, height),
    )
    own_overlap = bbox_iom(predicted, det.box)
    own_center = center_distance_norm(predicted, det.box, width, height)
    owner_overlaps = [
        bbox_iom(context.predicted_boxes[owner_id], det.box)
        for owner_id in owners
        if owner_id in context.predicted_boxes
    ]
    owner_centers = [
        center_distance_norm(context.predicted_boxes[owner_id], det.box, width, height)
        for owner_id in owners
        if owner_id in context.predicted_boxes
    ]
    best_owner_overlap = max(owner_overlaps, default=0.0)
    best_owner_center = min(owner_centers, default=1.0)

    clearly_better_than_owner = (
        own_overlap > best_owner_overlap + cfg.occlusion_competitor_margin
        and own_center < best_owner_center
    )
    return not clearly_better_than_owner


def track_detection_cost(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    raw_owner: dict[int, int],
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> float:
    if not low_conf_detection_is_plausible(track, det, width, height, cfg):
        return 1_000_000.0
    if detection_is_reserved_for_active_track(
        track,
        det,
        det_index,
        occlusion_context,
        width,
        height,
        cfg,
    ):
        return 1_000_000.0

    predicted = association_reference_box(track, det, width, height, cfg)
    iou_score = track_detection_overlap_score(track, predicted, det, cfg)
    center_cost = center_distance_norm(predicted, det.box, width, height)
    app_cost = hist_distance(track.mean_hist(), det.hist)
    area_cost = min(area_log_ratio(predicted, det.box), 2.0) / 2.0

    raw_penalty = 0.0
    if det.raw_id is not None:
        owner = raw_owner.get(det.raw_id)
        if owner is not None and owner != track.fixed_id:
            raw_penalty += 0.18
        elif track.top_raw_id() is not None and track.top_raw_id() != det.raw_id:
            raw_penalty += 0.05

    if track_is_lost_for_association(track):
        cost = (
            0.18 * (1.0 - iou_score)
            + 0.08 * min(center_cost, 1.0)
            + 0.52 * app_cost
            + 0.12 * area_cost
            + raw_penalty
        )
    else:
        cost = (
            0.42 * (1.0 - iou_score)
            + 0.22 * center_cost
            + 0.26 * app_cost
            + 0.10 * area_cost
            + raw_penalty
        )
    cost += occlusion_assignment_penalty(
        track,
        det,
        det_index,
        occlusion_context,
        width,
        height,
        cfg,
    )

    search_radius = 0.08 + min(track.missed, 60) / 60.0 * 0.22
    if (
        track.ever_detected
        and not track_is_lost_for_association(track)
        and iou_score < 0.01
        and center_cost > search_radius
    ):
        cost += 1.0
    return float(cost)


def association_cost_threshold(track: FixedTrack, cfg: TrackingConfig) -> float:
    if not track.ever_detected:
        return cfg.unseen_track_cost_threshold
    if track_is_lost_for_association(track):
        return cfg.lost_track_cost_threshold
    return cfg.match_cost_threshold


def match_and_update_tracks(
    tracks: dict[int, FixedTrack],
    detections: list[Detection],
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    cfg: TrackingConfig,
) -> None:
    from scipy.optimize import linear_sum_assignment

    height, width = frame.shape[:2]
    ordered_tracks = [tracks[idx] for idx in range(1, cfg.expected_pigs + 1)]
    raw_owner: dict[int, int] = {}
    for track in ordered_tracks:
        raw_id = track.top_raw_id()
        if raw_id is not None:
            raw_owner[raw_id] = track.fixed_id

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    occlusion_context = build_occlusion_context(
        ordered_tracks,
        detections,
        width,
        height,
        cfg,
    )

    def run_matching_phase(
        candidate_tracks: list[FixedTrack],
        detection_indices: list[int],
    ) -> None:
        if not candidate_tracks or not detection_indices:
            return

        costs = np.zeros(
            (len(candidate_tracks), len(detection_indices)),
            dtype=np.float32,
        )
        for row, track in enumerate(candidate_tracks):
            for col, det_idx in enumerate(detection_indices):
                costs[row, col] = track_detection_cost(
                    track,
                    detections[det_idx],
                    det_idx,
                    raw_owner,
                    occlusion_context,
                    width,
                    height,
                    cfg,
                )

        rows, cols = linear_sum_assignment(costs)
        for row, col in zip(rows, cols, strict=True):
            track = candidate_tracks[row]
            det_idx = detection_indices[col]
            if (
                track.fixed_id in matched_tracks
                or det_idx in matched_detections
                or costs[row, col] > association_cost_threshold(track, cfg)
            ):
                continue
            ambiguous = assignment_is_occlusion_ambiguous(
                track,
                det_idx,
                occlusion_context,
                cfg,
            )
            learn_identity = not (cfg.freeze_identity_in_occlusion and ambiguous)
            track.update_detected(
                detections[det_idx],
                width,
                height,
                cfg,
                learn_identity=learn_identity,
                ambiguous=ambiguous,
            )
            matched_tracks.add(track.fixed_id)
            matched_detections.add(det_idx)

    if detections:
        visible_tracks = [
            track for track in ordered_tracks if track_is_visible_for_association(track)
        ]
        reid_tracks = [
            track
            for track in ordered_tracks
            if not track_is_visible_for_association(track)
        ]
        all_detection_indices = list(range(len(detections)))

        run_matching_phase(visible_tracks, all_detection_indices)
        remaining_detection_indices = [
            idx for idx in all_detection_indices if idx not in matched_detections
        ]
        run_matching_phase(reid_tracks, remaining_detection_indices)

    # Unmatched high-confidence detections can initialize hidden placeholder IDs.
    unseen_tracks = [
        track
        for track in ordered_tracks
        if track.fixed_id not in matched_tracks and not track.ever_detected
    ]
    remaining_dets = [
        (idx, det)
        for idx, det in enumerate(detections)
        if idx not in matched_detections and det.score >= cfg.initial_track_conf
    ]
    for track, (det_idx, det) in zip(unseen_tracks, remaining_dets, strict=False):
        track.update_detected(det, width, height, cfg)
        matched_tracks.add(track.fixed_id)
        matched_detections.add(det_idx)

    for track in ordered_tracks:
        if track.fixed_id in matched_tracks:
            continue
        if should_hold_occluded_track_box(track, detections, occlusion_context, cfg):
            hold_box = track.hidden_motion_box(width, height, cfg)
            track.update_predicted(
                hold_box,
                width,
                height,
                ambiguous=True,
                hold=True,
            )
            continue

        lk_box = lk_predict_box(prev_frame, frame, track.last_box, width, height)
        if lk_box is None:
            lk_box = track.predicted_box(width, height)
        if track.missed > cfg.max_missing_frames:
            lk_box = 0.7 * track.last_box + 0.3 * lk_box
        track.update_predicted(lk_box, width, height)


# %%
def track_is_hidden(track: FixedTrack, cfg: TrackingConfig) -> bool:
    if not track.ever_detected:
        return True
    if (
        track.last_source == "occlusion_hold"
        and track.occlusion_hold_frames >= cfg.occlusion_hold_hidden_frames
    ):
        return True
    if track.missed >= cfg.hidden_missed_frames:
        return True
    return track.last_source == "predicted" and (
        track.last_score < cfg.hidden_score_threshold
    )


def shape_for_track(
    track: FixedTrack,
    frame_index: int,
    cfg: TrackingConfig,
) -> dict[str, Any]:
    hidden = "Yes" if track_is_hidden(track, cfg) else "No"
    needs_review = hidden == "Yes" or (
        track.last_source != "detected" or track.last_score < cfg.review_conf
    )
    x1, y1, x2, y2 = [round(float(value), 2) for value in track.last_box]
    return {
        "type": "rectangle",
        "occluded": hidden == "Yes",
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [x1, y1, x2, y2],
        "group": 0,
        "source": "file",
        "frame": int(frame_index),
        "attributes": [
            {"value": f"ID_{track.fixed_id}", "name": "ID"},
            {"value": cfg.default_behavior, "name": "Behavior"},
            {"value": hidden, "name": "Hidden"},
        ],
        "score": round(float(track.last_score), 4),
        "elements": [],
        "label": f"Pig_{track.fixed_id}",
        "_track_source": track.last_source,
        "_missed_frames": int(track.missed),
        "_needs_review": bool(needs_review),
        "_raw_track_id": track.top_raw_id(),
        "_ever_detected": bool(track.ever_detected),
        "_ambiguous_occlusion": bool(track.last_ambiguous),
        "_occlusion_hold": track.last_source == "occlusion_hold",
        "_motion_state": track.motion_state,
    }


def frame_shapes(
    tracks: dict[int, FixedTrack],
    frame_index: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    shapes = [
        shape_for_track(tracks[idx], frame_index, cfg)
        for idx in range(1, cfg.expected_pigs + 1)
    ]
    if len(shapes) != cfg.expected_pigs:
        raise RuntimeError(f"Expected {cfg.expected_pigs} shapes, got {len(shapes)}")
    return shapes


def write_annotation_json(path: Path, shapes: list[dict[str, Any]]) -> None:
    payload = [
        {
            "version": 0,
            "tags": [],
            "shapes": [strip_internal_shape_keys(shape) for shape in shapes],
            "tracks": [],
        }
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def strip_internal_shape_keys(shape: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in shape.items() if not key.startswith("_")}


def _shape_attributes_dict(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        str(attribute["name"]): attribute.get("value")
        for attribute in shape.get("attributes", [])
    }


def write_coco_annotation_json(
    path: Path,
    shapes: list[dict[str, Any]],
    video_path: Path,
    frame_width: int,
    frame_height: int,
    default_behavior: str,
    description: str = "Pig ID tracking annotations exported as COCO 1.0",
) -> None:
    """Write COCO 1.0 instances with Pig_N categories and CVAT attributes."""
    frames = sorted({int(shape["frame"]) for shape in shapes})
    image_id_by_frame = {frame: idx + 1 for idx, frame in enumerate(frames)}
    category_id_by_name = {f"Pig_{idx}": idx for idx in range(1, len(ID_VALUES) + 1)}

    images = [
        {
            "id": image_id_by_frame[frame],
            "file_name": f"frame_{frame:06d}.jpg",
            "width": int(frame_width),
            "height": int(frame_height),
            "frame": int(frame),
            "video": video_path.name,
        }
        for frame in frames
    ]

    annotations = []
    for annotation_id, shape in enumerate(shapes, start=1):
        x1, y1, x2, y2 = [float(value) for value in shape["points"]]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        label = str(shape["label"])
        attributes = _shape_attributes_dict(shape)
        fixed_id = int(label.removeprefix("Pig_"))
        attributes["track_id"] = fixed_id
        attributes["instance_id"] = fixed_id
        attributes.setdefault("Behavior", default_behavior)
        attributes.setdefault("Hidden", "No")
        attributes["TrackSource"] = str(shape.get("_track_source", "unknown"))
        attributes["NeedsReview"] = "Yes" if shape.get("_needs_review") else "No"
        attributes["Refined"] = "Yes" if shape.get("_refined") else "No"
        attributes["RefineReason"] = str(shape.get("_refine_reason", ""))
        attributes["MotionState"] = str(shape.get("_motion_state", "unknown"))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id_by_frame[int(shape["frame"])],
                "category_id": category_id_by_name[label],
                "track_id": fixed_id,
                "instance_id": fixed_id,
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(width, 2),
                    round(height, 2),
                ],
                "area": round(width * height, 2),
                "segmentation": [],
                "iscrowd": 0,
                "score": float(shape.get("score", 1.0)),
                "attributes": attributes,
            }
        )

    categories = [
        {
            "id": idx,
            "name": f"Pig_{idx}",
            "supercategory": "Pig",
            "attributes": {
                "ID": [f"ID_{idx}"],
                "Behavior": BEHAVIOR_VALUES,
                "Hidden": ["No", "Yes"],
            },
        }
        for idx in range(1, len(ID_VALUES) + 1)
    ]

    payload = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def shape_is_clean_for_training(shape: dict[str, Any], cfg: TrackingConfig) -> bool:
    attributes = _shape_attributes_dict(shape)
    return (
        shape.get("_track_source") == "detected"
        and attributes.get("Hidden", "No") == "No"
        and float(shape.get("score", 0.0)) >= cfg.review_conf
    )


def clean_training_shapes(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    return [shape for shape in shapes if shape_is_clean_for_training(shape, cfg)]


def shape_box(shape: dict[str, Any]) -> np.ndarray:
    return np.asarray(shape["points"], dtype=np.float32)


def set_shape_box(
    shape: dict[str, Any],
    box: np.ndarray,
    width: int,
    height: int,
) -> None:
    clipped = clip_box(box, width, height)
    shape["points"] = [round(float(value), 2) for value in clipped]


def shape_hidden_value(shape: dict[str, Any]) -> str:
    return str(_shape_attributes_dict(shape).get("Hidden", "No"))


def shape_is_stable_anchor(shape: dict[str, Any], cfg: TrackingConfig) -> bool:
    return (
        shape.get("_track_source") == "detected"
        and shape_hidden_value(shape) == "No"
        and float(shape.get("score", 0.0)) >= cfg.review_conf
    )


def interpolate_box(
    previous_shape: dict[str, Any],
    next_shape: dict[str, Any],
    target_frame: int,
) -> np.ndarray:
    previous_frame = int(previous_shape["frame"])
    next_frame = int(next_shape["frame"])
    if next_frame <= previous_frame:
        return shape_box(previous_shape)
    ratio = (target_frame - previous_frame) / float(next_frame - previous_frame)
    return (1.0 - ratio) * shape_box(previous_shape) + ratio * shape_box(next_shape)


def size_jump_ratio(box: np.ndarray, expected: np.ndarray) -> float:
    width, height = bbox_size(box)
    expected_width, expected_height = bbox_size(expected)
    width_ratio = abs(width / max(expected_width, 1e-6) - 1.0)
    height_ratio = abs(height / max(expected_height, 1e-6) - 1.0)
    return max(width_ratio, height_ratio)


def nearby_anchor_indices(
    track_shapes: list[dict[str, Any]],
    stable_indices: list[int],
    current_index: int,
    cfg: TrackingConfig,
) -> tuple[int | None, int | None]:
    frame = int(track_shapes[current_index]["frame"])
    previous_idx = None
    next_idx = None
    for idx in reversed(stable_indices):
        if idx >= current_index:
            continue
        if frame - int(track_shapes[idx]["frame"]) <= cfg.refine_max_gap_frames:
            previous_idx = idx
        break
    for idx in stable_indices:
        if idx <= current_index:
            continue
        if int(track_shapes[idx]["frame"]) - frame <= cfg.refine_max_gap_frames:
            next_idx = idx
        break
    return previous_idx, next_idx


def refine_original_weight(shape: dict[str, Any], cfg: TrackingConfig) -> float:
    if shape.get("_track_source") != "detected":
        return 0.15
    if float(shape.get("score", 0.0)) < cfg.review_conf:
        return 0.35
    return 0.65


def refine_shapes_temporally(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Use before/after stable boxes to reduce one-frame bbox jumps."""
    if not cfg.refine_boxes:
        return shapes

    refined_shapes = [shape.copy() for shape in shapes]
    for shape in refined_shapes:
        shape["_refined"] = False
        shape["_refine_reason"] = ""

    for fixed_id in range(1, cfg.expected_pigs + 1):
        track_shapes = sorted(
            [
                shape
                for shape in refined_shapes
                if str(shape["label"]) == f"Pig_{fixed_id}"
            ],
            key=lambda item: int(item["frame"]),
        )
        stable_indices = [
            idx
            for idx, shape in enumerate(track_shapes)
            if shape_is_stable_anchor(shape, cfg)
        ]
        if not stable_indices:
            continue

        for idx, shape in enumerate(track_shapes):
            frame = int(shape["frame"])
            previous_idx, next_idx = nearby_anchor_indices(
                track_shapes,
                stable_indices,
                idx,
                cfg,
            )
            if previous_idx is None and next_idx is None:
                continue

            original = shape_box(shape)
            expected = None
            if previous_idx is not None and next_idx is not None:
                expected = interpolate_box(
                    track_shapes[previous_idx],
                    track_shapes[next_idx],
                    frame,
                )
            elif not shape_is_stable_anchor(shape, cfg):
                anchor_idx = previous_idx if previous_idx is not None else next_idx
                if anchor_idx is not None:
                    expected = shape_box(track_shapes[anchor_idx])
            if expected is None:
                continue

            source = str(shape.get("_track_source", "unknown"))
            unstable_detection = (
                source != "detected"
                or shape_hidden_value(shape) == "Yes"
                or float(shape.get("score", 0.0)) < cfg.review_conf
            )
            size_jump = size_jump_ratio(original, expected)
            size_outlier = size_jump > cfg.refine_size_jump_threshold
            if not unstable_detection and not size_outlier:
                continue

            original_weight = refine_original_weight(shape, cfg)
            if size_outlier:
                original_weight = min(original_weight, 0.35)
            if size_outlier and not unstable_detection:
                reason = f"size_jump>{cfg.refine_size_jump_threshold:.2f}"
            elif source != "detected":
                reason = source
            else:
                reason = "low_score_or_hidden"
            refined = original_weight * original + (1.0 - original_weight) * expected
            shape["_original_points"] = [round(float(value), 2) for value in original]
            shape["_refined"] = True
            shape["_refine_reason"] = reason
            shape["_refine_size_jump"] = round(float(size_jump), 4)
            set_shape_box(shape, refined, width, height)

    return refined_shapes


def shape_fixed_id(shape: dict[str, Any]) -> int:
    return int(str(shape["label"]).removeprefix("Pig_"))


def transition_cost(
    previous_box: np.ndarray,
    current_box: np.ndarray,
    width: int,
    height: int,
) -> float:
    center_cost = center_distance_norm(previous_box, current_box, width, height)
    size_cost = min(area_log_ratio(previous_box, current_box), 2.0) / 2.0
    return float(center_cost + 0.10 * size_cost)


def identity_swap_reason(
    previous_first: dict[str, Any],
    previous_second: dict[str, Any],
    current_first: dict[str, Any],
    current_second: dict[str, Any],
    gain: float,
    cfg: TrackingConfig,
) -> str | None:
    previous_iom = bbox_iom(shape_box(previous_first), shape_box(previous_second))
    current_iom = bbox_iom(shape_box(current_first), shape_box(current_second))
    if max(previous_iom, current_iom) >= cfg.identity_swap_iom_threshold:
        return "overlap_continuity"

    uncertain = any(
        shape.get("_needs_review")
        or shape.get("_ambiguous_occlusion")
        or shape.get("_occlusion_hold")
        or shape.get("_track_source") != "detected"
        for shape in (current_first, current_second)
    )
    if uncertain:
        return "review_continuity"
    if gain >= cfg.identity_swap_min_gain * 2.0:
        return "large_continuity_gain"
    return None


def _non_id_attribute_values(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        str(attribute["name"]): attribute.get("value")
        for attribute in shape.get("attributes", [])
        if str(attribute.get("name")) != "ID"
    }


def _apply_non_id_attribute_values(
    shape: dict[str, Any],
    values: dict[str, Any],
) -> None:
    for attribute in shape.get("attributes", []):
        name = str(attribute.get("name"))
        if name != "ID" and name in values:
            attribute["value"] = values[name]


def _shape_payload(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in shape.items()
        if key not in {"label", "attributes", "elements"}
    }


def _apply_shape_payload(
    shape: dict[str, Any],
    payload: dict[str, Any],
    attributes: dict[str, Any],
) -> None:
    for key, value in payload.items():
        shape[key] = value
    _apply_non_id_attribute_values(shape, attributes)


def swap_shape_identity_payloads(
    first: dict[str, Any],
    second: dict[str, Any],
    reason: str,
) -> None:
    first_id = shape_fixed_id(first)
    second_id = shape_fixed_id(second)
    first_payload = _shape_payload(first)
    second_payload = _shape_payload(second)
    first_attrs = _non_id_attribute_values(first)
    second_attrs = _non_id_attribute_values(second)

    _apply_shape_payload(first, second_payload, second_attrs)
    _apply_shape_payload(second, first_payload, first_attrs)
    for shape, other_id in ((first, second_id), (second, first_id)):
        shape["_identity_swap_guard"] = True
        shape["_identity_swap_with"] = int(other_id)
        shape["_identity_swap_reason"] = reason
        shape["_needs_review"] = True


def apply_identity_swap_guard(
    shapes: list[dict[str, Any]],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    """Swap per-frame geometry back when a pair assignment breaks continuity."""
    if not cfg.identity_swap_guard:
        return shapes

    guarded_shapes = [shape.copy() for shape in shapes]
    frames = sorted({int(shape["frame"]) for shape in guarded_shapes})
    previous_by_id: dict[int, dict[str, Any]] | None = None

    for frame in frames:
        current_by_id = {
            shape_fixed_id(shape): shape
            for shape in guarded_shapes
            if int(shape["frame"]) == frame
        }
        if previous_by_id is None:
            previous_by_id = {
                fixed_id: shape.copy()
                for fixed_id, shape in current_by_id.items()
            }
            continue

        changed = True
        while changed:
            changed = False
            best_pair: tuple[int, int] | None = None
            best_gain = cfg.identity_swap_min_gain
            best_reason: str | None = None
            ids = sorted(set(previous_by_id).intersection(current_by_id))

            for idx, first_id in enumerate(ids):
                for second_id in ids[idx + 1 :]:
                    prev_first = previous_by_id[first_id]
                    prev_second = previous_by_id[second_id]
                    cur_first = current_by_id[first_id]
                    cur_second = current_by_id[second_id]
                    own_cost = transition_cost(
                        shape_box(prev_first),
                        shape_box(cur_first),
                        width,
                        height,
                    ) + transition_cost(
                        shape_box(prev_second),
                        shape_box(cur_second),
                        width,
                        height,
                    )
                    swapped_cost = transition_cost(
                        shape_box(prev_first),
                        shape_box(cur_second),
                        width,
                        height,
                    ) + transition_cost(
                        shape_box(prev_second),
                        shape_box(cur_first),
                        width,
                        height,
                    )
                    gain = own_cost - swapped_cost
                    if gain <= best_gain:
                        continue
                    reason = identity_swap_reason(
                        prev_first,
                        prev_second,
                        cur_first,
                        cur_second,
                        gain,
                        cfg,
                    )
                    if reason is None:
                        continue
                    best_pair = (first_id, second_id)
                    best_gain = gain
                    best_reason = reason

            if best_pair is not None and best_reason is not None:
                first_id, second_id = best_pair
                swap_shape_identity_payloads(
                    current_by_id[first_id],
                    current_by_id[second_id],
                    best_reason,
                )
                changed = True

        previous_by_id = {
            fixed_id: shape.copy()
            for fixed_id, shape in current_by_id.items()
        }

    return guarded_shapes


def _shape_attribute_value(
    shape: dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    return _shape_attributes_dict(shape).get(name, default)


def build_quality_report(
    shapes: list[dict[str, Any]],
    cfg: TrackingConfig,
    video_path: Path,
    source_fps: float,
    source_frame_count: int,
) -> dict[str, Any]:
    """Summarize frames/tracks that need manual review."""
    frames = sorted({int(shape["frame"]) for shape in shapes})
    shapes_by_frame = {
        frame: [shape for shape in shapes if int(shape["frame"]) == frame]
        for frame in frames
    }
    frame_rows: list[dict[str, Any]] = []
    issue_frames: list[int] = []

    for frame in frames:
        frame_shapes = shapes_by_frame[frame]
        hidden_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
        ]
        predicted_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_track_source") != "detected"
        ]
        refined_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_refined")
        ]
        ambiguous_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_ambiguous_occlusion")
        ]
        hold_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_occlusion_hold")
        ]
        identity_swap_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_identity_swap_guard")
        ]
        moving_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state") == "moving"
        ]
        stationary_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state") == "stationary"
        ]
        unknown_motion_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if shape.get("_motion_state", "unknown") == "unknown"
        ]
        low_score_ids = [
            int(str(shape["label"]).removeprefix("Pig_"))
            for shape in frame_shapes
            if float(shape.get("score", 0.0)) < cfg.review_conf
        ]
        review_ids = sorted(
            set(
                hidden_ids
                + predicted_ids
                + low_score_ids
                + refined_ids
                + ambiguous_ids
                + hold_ids
                + identity_swap_ids
            )
        )
        detected_count = sum(
            1 for shape in frame_shapes if shape.get("_track_source") == "detected"
        )
        min_score = min(
            (float(shape.get("score", 0.0)) for shape in frame_shapes),
            default=0.0,
        )
        row = {
            "frame": frame,
            "time_sec": round(frame / max(source_fps, 1e-6), 3),
            "shape_count": len(frame_shapes),
            "detected_count": detected_count,
            "predicted_count": len(predicted_ids),
            "refined_count": len(refined_ids),
            "ambiguous_occlusion_count": len(ambiguous_ids),
            "occlusion_hold_count": len(hold_ids),
            "identity_swap_guard_count": len(identity_swap_ids),
            "hidden_count": len(hidden_ids),
            "low_score_count": len(low_score_ids),
            "min_score": round(min_score, 4),
            "hidden_ids": hidden_ids,
            "predicted_ids": predicted_ids,
            "refined_ids": refined_ids,
            "ambiguous_occlusion_ids": ambiguous_ids,
            "occlusion_hold_ids": hold_ids,
            "identity_swap_guard_ids": identity_swap_ids,
            "moving_ids": moving_ids,
            "stationary_ids": stationary_ids,
            "unknown_motion_ids": unknown_motion_ids,
            "low_score_ids": low_score_ids,
            "review_ids": review_ids,
            "needs_review": bool(review_ids or len(frame_shapes) != cfg.expected_pigs),
        }
        if row["needs_review"]:
            issue_frames.append(frame)
        frame_rows.append(row)

    track_rows: list[dict[str, Any]] = []
    for fixed_id in range(1, cfg.expected_pigs + 1):
        track_shapes = [
            shape
            for shape in shapes
            if str(shape["label"]) == f"Pig_{fixed_id}"
        ]
        scores = [float(shape.get("score", 0.0)) for shape in track_shapes]
        detected_frames = sum(
            1 for shape in track_shapes if shape.get("_track_source") == "detected"
        )
        predicted_frames = len(track_shapes) - detected_frames
        hidden_frames = sum(
            1
            for shape in track_shapes
            if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
        )
        refined_frames = sum(1 for shape in track_shapes if shape.get("_refined"))
        ambiguous_frames = sum(
            1 for shape in track_shapes if shape.get("_ambiguous_occlusion")
        )
        hold_frames = sum(1 for shape in track_shapes if shape.get("_occlusion_hold"))
        identity_swap_frames = sum(
            1 for shape in track_shapes if shape.get("_identity_swap_guard")
        )
        moving_frames = sum(
            1 for shape in track_shapes if shape.get("_motion_state") == "moving"
        )
        stationary_frames = sum(
            1 for shape in track_shapes if shape.get("_motion_state") == "stationary"
        )
        unknown_motion_frames = sum(
            1
            for shape in track_shapes
            if shape.get("_motion_state", "unknown") == "unknown"
        )
        review_frames = sum(
            1
            for shape in track_shapes
            if (
                shape.get("_needs_review")
                or shape.get("_refined")
                or shape.get("_ambiguous_occlusion")
                or shape.get("_occlusion_hold")
                or shape.get("_identity_swap_guard")
            )
        )
        track_rows.append(
            {
                "fixed_id": fixed_id,
                "label": f"Pig_{fixed_id}",
                "id_attribute": f"ID_{fixed_id}",
                "frames": len(track_shapes),
                "detected_frames": detected_frames,
                "predicted_frames": predicted_frames,
                "refined_frames": refined_frames,
                "ambiguous_occlusion_frames": ambiguous_frames,
                "occlusion_hold_frames": hold_frames,
                "identity_swap_guard_frames": identity_swap_frames,
                "moving_frames": moving_frames,
                "stationary_frames": stationary_frames,
                "unknown_motion_frames": unknown_motion_frames,
                "hidden_frames": hidden_frames,
                "review_frames": review_frames,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            }
        )

    clean_shape_count = len(clean_training_shapes(shapes, cfg))
    return {
        "video": str(video_path),
        "video_name": video_path.name,
        "source_fps": round(float(source_fps), 4),
        "source_frame_count": int(source_frame_count),
        "start_frame": int(cfg.start_frame),
        "processed_frames": len(frames),
        "expected_pigs": int(cfg.expected_pigs),
        "thresholds": {
            "det_conf": cfg.det_conf,
            "track_high_conf": cfg.track_high_conf,
            "review_conf": cfg.review_conf,
            "iou": cfg.iou,
            "use_mask_iou": cfg.use_mask_iou,
            "mask_iou_max_missed": cfg.mask_iou_max_missed,
            "mask_iou_min_area": cfg.mask_iou_min_area,
            "match_cost_threshold": cfg.match_cost_threshold,
            "unseen_track_cost_threshold": cfg.unseen_track_cost_threshold,
            "lost_track_cost_threshold": cfg.lost_track_cost_threshold,
            "lost_track_reid_appearance_threshold": (
                cfg.lost_track_reid_appearance_threshold
            ),
            "initial_track_conf": cfg.initial_track_conf,
            "motion_gate_confidence": cfg.motion_gate_confidence,
            "low_conf_motion_gate": cfg.low_conf_motion_gate,
            "low_conf_max_center_jump": cfg.low_conf_max_center_jump,
            "low_conf_max_box_jump_scale": cfg.low_conf_max_box_jump_scale,
            "low_conf_min_iou": cfg.low_conf_min_iou,
            "occlusion_aware_matching": cfg.occlusion_aware_matching,
            "occlusion_track_iom_threshold": cfg.occlusion_track_iom_threshold,
            "occlusion_detection_iom_threshold": (
                cfg.occlusion_detection_iom_threshold
            ),
            "occlusion_stationary_speed": cfg.occlusion_stationary_speed,
            "occlusion_stationary_max_center_jump": (
                cfg.occlusion_stationary_max_center_jump
            ),
            "occlusion_switch_penalty": cfg.occlusion_switch_penalty,
            "occlusion_competitor_margin": cfg.occlusion_competitor_margin,
            "occlusion_appearance_penalty": cfg.occlusion_appearance_penalty,
            "occlusion_appearance_margin": cfg.occlusion_appearance_margin,
            "occlusion_stationary_lock": cfg.occlusion_stationary_lock,
            "freeze_identity_in_occlusion": cfg.freeze_identity_in_occlusion,
            "hold_occluded_box": cfg.hold_occluded_box,
            "occlusion_hold_max_frames": cfg.occlusion_hold_max_frames,
            "occlusion_hold_hidden_frames": cfg.occlusion_hold_hidden_frames,
            "identity_swap_guard": cfg.identity_swap_guard,
            "identity_swap_min_gain": cfg.identity_swap_min_gain,
            "identity_swap_iom_threshold": cfg.identity_swap_iom_threshold,
            "hidden_motion_model": cfg.hidden_motion_model,
            "hidden_velocity_alpha": cfg.hidden_velocity_alpha,
            "hidden_acceleration_alpha": cfg.hidden_acceleration_alpha,
            "hidden_stationary_speed": cfg.hidden_stationary_speed,
            "hidden_motion_history": cfg.hidden_motion_history,
            "hidden_min_motion_history": cfg.hidden_min_motion_history,
            "hidden_stationary_displacement": cfg.hidden_stationary_displacement,
            "hidden_moving_displacement": cfg.hidden_moving_displacement,
            "hidden_motion_consistency": cfg.hidden_motion_consistency,
            "hidden_stationary_lock_frames": cfg.hidden_stationary_lock_frames,
            "hidden_max_motion_step_box_scale": cfg.hidden_max_motion_step_box_scale,
        },
        "summary": {
            "total_shapes": len(shapes),
            "clean_training_shapes": clean_shape_count,
            "review_shapes": sum(
                1
                for shape in shapes
                if shape.get("_needs_review") or shape.get("_refined")
            ),
            "refined_shapes": sum(1 for shape in shapes if shape.get("_refined")),
            "ambiguous_occlusion_shapes": sum(
                1 for shape in shapes if shape.get("_ambiguous_occlusion")
            ),
            "occlusion_hold_shapes": sum(
                1 for shape in shapes if shape.get("_occlusion_hold")
            ),
            "identity_swap_guard_shapes": sum(
                1 for shape in shapes if shape.get("_identity_swap_guard")
            ),
            "hidden_shapes": sum(
                1
                for shape in shapes
                if _shape_attribute_value(shape, "Hidden", "No") == "Yes"
            ),
            "issue_frame_count": len(issue_frames),
            "issue_frames": issue_frames,
        },
        "frames": frame_rows,
        "tracks": track_rows,
    }


def write_quality_report_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def write_quality_report_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "frame",
        "time_sec",
        "shape_count",
        "detected_count",
        "predicted_count",
        "refined_count",
        "ambiguous_occlusion_count",
        "occlusion_hold_count",
        "identity_swap_guard_count",
        "hidden_count",
        "low_score_count",
        "min_score",
        "hidden_ids",
        "predicted_ids",
        "refined_ids",
        "ambiguous_occlusion_ids",
        "occlusion_hold_ids",
        "identity_swap_guard_ids",
        "moving_ids",
        "stationary_ids",
        "unknown_motion_ids",
        "low_score_ids",
        "review_ids",
        "needs_review",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["frames"]:
            serialized = row.copy()
            for key in (
                "hidden_ids",
                "predicted_ids",
                "refined_ids",
                "ambiguous_occlusion_ids",
                "occlusion_hold_ids",
                "identity_swap_guard_ids",
                "moving_ids",
                "stationary_ids",
                "unknown_motion_ids",
                "low_score_ids",
                "review_ids",
            ):
                serialized[key] = " ".join(str(value) for value in row[key])
            writer.writerow(serialized)


def _xml_child(parent: ET.Element, tag: str, text: Any = "") -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def _append_cvat_xml_label(parent: ET.Element, label: dict[str, Any]) -> None:
    label_el = ET.SubElement(parent, "label")
    _xml_child(label_el, "name", label["name"])
    _xml_child(label_el, "type", label.get("type", "any"))
    attrs_el = ET.SubElement(label_el, "attributes")
    for attribute in label.get("attributes", []):
        attr_el = ET.SubElement(attrs_el, "attribute")
        _xml_child(attr_el, "name", attribute["name"])
        _xml_child(attr_el, "mutable", str(bool(attribute["mutable"])))
        _xml_child(attr_el, "input_type", attribute["input_type"])
        _xml_child(attr_el, "default_value", attribute["default_value"])
        _xml_child(attr_el, "values", "\n".join(attribute.get("values", [])))


def write_cvat_video_xml(
    path: Path,
    shapes: list[dict[str, Any]],
    video_path: Path,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> None:
    """Write native CVAT for video 1.1 XML with real track elements."""
    root = ET.Element("annotations")
    _xml_child(root, "version", "1.1")

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    _xml_child(task, "id", 0)
    _xml_child(task, "name", video_path.stem)
    _xml_child(task, "size", frame_count)
    _xml_child(task, "mode", "interpolation")
    _xml_child(task, "overlap", 0)
    _xml_child(task, "bugtracker", "")
    _xml_child(task, "flipped", "False")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _xml_child(task, "created", now)
    _xml_child(task, "updated", now)

    labels_el = ET.SubElement(task, "labels")
    for label in PIG_LABEL_SCHEMA:
        _append_cvat_xml_label(labels_el, label)

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    _xml_child(segment, "id", 0)
    _xml_child(segment, "start", 0)
    _xml_child(segment, "stop", max(0, frame_count - 1))
    _xml_child(segment, "url", "")

    owner = ET.SubElement(task, "owner")
    _xml_child(owner, "username", "auto")
    _xml_child(owner, "email", "")

    original_size = ET.SubElement(task, "original_size")
    _xml_child(original_size, "width", int(frame_width))
    _xml_child(original_size, "height", int(frame_height))
    _xml_child(meta, "dumped", now)

    shapes_by_track: dict[int, list[dict[str, Any]]] = {
        fixed_id: [] for fixed_id in range(1, len(ID_VALUES) + 1)
    }
    for shape in shapes:
        fixed_id = int(str(shape["label"]).removeprefix("Pig_"))
        shapes_by_track[fixed_id].append(shape)

    for fixed_id in range(1, len(ID_VALUES) + 1):
        track = ET.SubElement(
            root,
            "track",
            {
                "id": str(fixed_id),
                "label": f"Pig_{fixed_id}",
                "source": "auto",
            },
        )
        for shape in sorted(shapes_by_track[fixed_id], key=lambda item: item["frame"]):
            x1, y1, x2, y2 = [float(value) for value in shape["points"]]
            attributes = _shape_attributes_dict(shape)
            hidden = str(attributes.get("Hidden", "No"))
            box = ET.SubElement(
                track,
                "box",
                {
                    "frame": str(int(shape["frame"])),
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "outside": "0",
                    "occluded": "1" if hidden == "Yes" else "0",
                    "keyframe": "1",
                },
            )
            for name in ("ID", "Behavior", "Hidden"):
                _xml_child(box, "attribute", attributes.get(name, "")).set(
                    "name",
                    name,
                )

    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(
        indent="  ",
        encoding="utf-8",
    )
    path.write_bytes(pretty_xml)


def write_labels_json(path: Path) -> None:
    path.write_text(
        json.dumps(PIG_LABEL_SCHEMA, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


# %%
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


# %%
def run_tracking(cfg: TrackingConfig) -> TrackingSummary:
    """Run YOLOv8 + mask + stabilized eight-ID tracking."""
    validate_config(cfg)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Install tracking dependencies first: pip install -e .[tracking]"
        ) from exc

    (
        output_video,
        annotations_json,
        coco_annotations_json,
        clean_coco_annotations_json,
        cvat_video_xml,
        labels_json,
        tracker_yaml,
        quality_report_json,
        quality_report_csv,
    ) = resolve_output_paths(cfg)
    write_tracker_yaml(tracker_yaml, cfg)
    write_labels_json(labels_json)

    capture = cv2.VideoCapture(str(cfg.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or cfg.output_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video frame size.")
    if total_frames and cfg.start_frame >= total_frames:
        raise ValueError(
            f"start_frame={cfg.start_frame} is outside video with "
            f"{total_frames} frames."
        )
    if cfg.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    mask = load_mask(cfg.mask_path, width, height, cfg)
    model = YOLO(str(cfg.weights_path))
    tracks: dict[int, FixedTrack] | None = None
    shapes: list[dict[str, Any]] = []
    hidden_shape_count = 0
    review_shape_count = 0
    frame_index = cfg.start_frame - 1
    frames_written = 0
    prev_frame: np.ndarray | None = None
    show_enabled = cfg.show

    try:
        from tqdm import tqdm

        remaining_frames = (
            max(0, total_frames - cfg.start_frame) if total_frames else None
        )
        progress_total = cfg.max_frames
        if progress_total is None:
            progress_total = remaining_frames
        elif remaining_frames is not None:
            progress_total = min(progress_total, remaining_frames)
        progress = tqdm(
            total=progress_total,
            desc="Tracking 8 pigs",
        )
    except ImportError:
        progress = None

    try:
        while True:
            if cfg.max_frames is not None and frames_written >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            frame_h, frame_w = frame.shape[:2]
            if frame_w != width or frame_h != height:
                width, height = frame_w, frame_h
                mask = load_mask(cfg.mask_path, width, height, cfg)

            detector_frame = (
                apply_mask_to_frame(frame, mask)
                if cfg.mask_input_frame and mask is not None
                else frame
            )
            results = model.track(
                source=detector_frame,
                persist=True,
                conf=cfg.det_conf,
                iou=cfg.iou,
                tracker=str(tracker_yaml),
                verbose=False,
            )
            detections = adaptive_confidence_filter(
                parse_detections(results[0], frame, mask, cfg),
                cfg,
            )

            if tracks is None:
                tracks = initialize_tracks(detections, mask, width, height, cfg)
            else:
                match_and_update_tracks(tracks, detections, frame, prev_frame, cfg)

            current_shapes = frame_shapes(tracks, frame_index, cfg)
            shapes.extend(current_shapes)
            frames_written += 1
            prev_frame = frame.copy()

            if progress is not None:
                progress.update(1)

            if show_enabled:
                try:
                    annotated = draw_tracks(frame, tracks, mask, frame_index, cfg)
                    cv2.imshow("Pig ID tracking", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
                except cv2.error:
                    print("OpenCV GUI preview is unavailable; continuing headless.")
                    show_enabled = False
    finally:
        capture.release()
        if progress is not None:
            progress.close()
        if show_enabled:
            cv2.destroyAllWindows()

    if frames_written == 0:
        raise RuntimeError("No frames were processed.")

    shapes = apply_identity_swap_guard(shapes, width, height, cfg)
    shapes = refine_shapes_temporally(shapes, width, height, cfg)
    hidden_shape_count = sum(
        1 for shape in shapes if shape_hidden_value(shape) == "Yes"
    )
    review_shape_count = sum(
        1
        for shape in shapes
        if (
            shape.get("_needs_review")
            or shape.get("_refined")
            or shape.get("_ambiguous_occlusion")
            or shape.get("_occlusion_hold")
            or shape.get("_identity_swap_guard")
        )
    )
    rendered_frames = render_annotation_video(
        cfg.video_path,
        output_video,
        shapes,
        cfg,
        frame_limit=frames_written,
    )
    if rendered_frames != frames_written:
        raise RuntimeError(
            f"Rendered {rendered_frames} frames, but tracked {frames_written} frames."
        )

    write_annotation_json(annotations_json, shapes)
    write_coco_annotation_json(
        coco_annotations_json,
        shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior,
    )
    clean_shapes = clean_training_shapes(shapes, cfg)
    write_coco_annotation_json(
        clean_coco_annotations_json,
        clean_shapes,
        cfg.video_path,
        width,
        height,
        cfg.default_behavior,
        description=(
            "Clean pig training annotations exported as COCO 1.0 "
            "from detected, non-hidden, high-confidence boxes only"
        ),
    )
    max_shape_frame = max(int(shape["frame"]) for shape in shapes)
    source_frame_count = max(total_frames, max_shape_frame + 1)
    write_cvat_video_xml(
        cvat_video_xml,
        shapes,
        cfg.video_path,
        width,
        height,
        source_frame_count,
    )
    quality_report = build_quality_report(
        shapes,
        cfg,
        cfg.video_path,
        source_fps,
        source_frame_count,
    )
    write_quality_report_json(quality_report_json, quality_report)
    write_quality_report_csv(quality_report_csv, quality_report)
    return TrackingSummary(
        output_video=output_video,
        annotations_json=annotations_json,
        coco_annotations_json=coco_annotations_json,
        clean_coco_annotations_json=clean_coco_annotations_json,
        cvat_video_xml=cvat_video_xml,
        labels_json=labels_json,
        quality_report_json=quality_report_json,
        quality_report_csv=quality_report_csv,
        frames_read=frames_written,
        frames_written=frames_written,
        shape_count=len(shapes),
        hidden_shape_count=hidden_shape_count,
        review_shape_count=review_shape_count,
        start_frame=cfg.start_frame,
        source_fps=source_fps,
        output_fps=cfg.output_fps,
    )


def display_tracked_video(
    video_path: Path,
    width: int = 900,
    embed: bool = False,
) -> None:
    """Display the tracked MP4 directly in a notebook output cell."""
    from IPython.display import Video, display

    display(Video(str(video_path), embed=embed, width=width))


# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-key", type=str, default=None)
    parser.add_argument(
        "--all-config-videos",
        action="store_true",
        help="Run every video listed in the selected tracking path profile.",
    )
    parser.add_argument(
        "--path-config",
        type=Path,
        default=DEFAULT_TRACKING_PATH_CONFIG,
        help="JSON path profile file for fast video/weights/mask/output switching.",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--coco-json", type=Path, default=None)
    parser.add_argument("--clean-coco-json", type=Path, default=None)
    parser.add_argument("--cvat-video-xml", type=Path, default=None)
    parser.add_argument("--labels-json", type=Path, default=None)
    parser.add_argument("--tracker-yaml", type=Path, default=None)
    parser.add_argument("--quality-report-json", type=Path, default=None)
    parser.add_argument("--quality-report-csv", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Deprecated alias for --review-conf.",
    )
    parser.add_argument("--det-conf", type=float, default=DEFAULT_DET_CONF_THRESHOLD)
    parser.add_argument(
        "--track-high-conf",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument(
        "--review-conf",
        type=float,
        default=DEFAULT_REVIEW_CONF_THRESHOLD,
    )
    parser.add_argument("--adaptive-conf-step", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--default-behavior", type=str, default="lying")
    parser.add_argument("--roi-mode", choices=["center", "cover"], default="center")
    parser.add_argument("--roi-min-cover", type=float, default=0.10)
    parser.add_argument("--roi-dilate-px", type=int, default=8)
    parser.add_argument("--hidden-missed-frames", type=int, default=5)
    parser.add_argument("--hidden-score-threshold", type=float, default=0.15)
    parser.add_argument("--mask-iou-max-missed", type=int, default=10)
    parser.add_argument("--mask-iou-min-area", type=int, default=64)
    parser.add_argument("--max-missing-frames", type=int, default=90)
    parser.add_argument("--match-cost-threshold", type=float, default=0.78)
    parser.add_argument("--unseen-track-cost-threshold", type=float, default=1.10)
    parser.add_argument("--lost-track-cost-threshold", type=float, default=0.95)
    parser.add_argument(
        "--lost-track-reid-appearance-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--initial-track-conf",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument(
        "--motion-gate-confidence",
        type=float,
        default=DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    )
    parser.add_argument("--low-conf-max-center-jump", type=float, default=0.08)
    parser.add_argument("--low-conf-max-box-jump-scale", type=float, default=1.75)
    parser.add_argument("--low-conf-min-iou", type=float, default=0.01)
    parser.add_argument("--occlusion-track-iom-threshold", type=float, default=0.20)
    parser.add_argument("--occlusion-detection-iom-threshold", type=float, default=0.30)
    parser.add_argument("--occlusion-stationary-speed", type=float, default=0.006)
    parser.add_argument(
        "--occlusion-stationary-max-center-jump",
        type=float,
        default=0.045,
    )
    parser.add_argument("--occlusion-switch-penalty", type=float, default=0.45)
    parser.add_argument("--occlusion-competitor-margin", type=float, default=0.12)
    parser.add_argument("--occlusion-appearance-penalty", type=float, default=0.30)
    parser.add_argument("--occlusion-appearance-margin", type=float, default=0.08)
    parser.add_argument("--occlusion-hold-max-frames", type=int, default=30)
    parser.add_argument("--occlusion-hold-hidden-frames", type=int, default=2)
    parser.add_argument("--identity-swap-min-gain", type=float, default=0.015)
    parser.add_argument("--identity-swap-iom-threshold", type=float, default=0.10)
    parser.add_argument("--hidden-velocity-alpha", type=float, default=0.65)
    parser.add_argument("--hidden-acceleration-alpha", type=float, default=0.35)
    parser.add_argument("--hidden-stationary-speed", type=float, default=0.006)
    parser.add_argument("--hidden-motion-history", type=int, default=8)
    parser.add_argument("--hidden-min-motion-history", type=int, default=4)
    parser.add_argument("--hidden-stationary-displacement", type=float, default=0.015)
    parser.add_argument("--hidden-moving-displacement", type=float, default=0.035)
    parser.add_argument("--hidden-motion-consistency", type=float, default=0.55)
    parser.add_argument("--hidden-stationary-lock-frames", type=int, default=8)
    parser.add_argument("--hidden-max-motion-step-box-scale", type=float, default=1.50)
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
    )
    parser.add_argument("--max-box-scale-change", type=float, default=0.25)
    parser.add_argument("--max-box-scale-change-after-gap", type=float, default=0.75)
    parser.add_argument("--high-conf-smooth-alpha", type=float, default=0.75)
    parser.add_argument("--mid-conf-smooth-alpha", type=float, default=0.55)
    parser.add_argument("--low-conf-smooth-alpha", type=float, default=0.35)
    parser.add_argument("--refine-max-gap", type=int, default=15)
    parser.add_argument("--refine-size-jump-threshold", type=float, default=0.45)
    parser.add_argument("--visual-opacity", type=float, default=DEFAULT_VISUAL_OPACITY)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-mask-input", action="store_true")
    parser.add_argument("--no-mask-iou", action="store_true")
    parser.add_argument("--no-smooth-boxes", action="store_true")
    parser.add_argument("--no-refine-boxes", action="store_true")
    parser.add_argument("--no-low-conf-motion-gate", action="store_true")
    parser.add_argument("--no-occlusion-aware-matching", action="store_true")
    parser.add_argument("--learn-identity-in-occlusion", action="store_true")
    parser.add_argument("--no-hold-occluded-box", action="store_true")
    parser.add_argument("--no-identity-swap-guard", action="store_true")
    parser.add_argument("--no-occlusion-stationary-lock", action="store_true")
    parser.add_argument("--no-hidden-motion-model", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--display-inline", action="store_true")
    return parser.parse_args(argv)


def _profile_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_tracking_path_profile(args.path_config, args.profile)


def _tracking_config_from_args(
    args: argparse.Namespace,
    profile: dict[str, Any],
    video_path: Path | None = None,
) -> TrackingConfig:
    selected_video = (
        video_path
        or args.video
        or profile_video_path(profile, args.video_key, DEFAULT_VIDEO_PATH)
        or DEFAULT_VIDEO_PATH
    )
    weights_path = (
        args.weights
        or profile_path(profile, "weights", DEFAULT_WEIGHTS_PATH)
        or DEFAULT_WEIGHTS_PATH
    )
    mask_path = None
    if not args.no_mask:
        mask_path = (
            args.mask
            or profile_path(profile, "mask", DEFAULT_MASK_PATH)
            or DEFAULT_MASK_PATH
        )
    output_dir = (
        args.output_dir
        or profile_path(profile, "output_dir", DEFAULT_OUTPUT_DIR)
        or DEFAULT_OUTPUT_DIR
    )
    return TrackingConfig(
        video_path=selected_video,
        weights_path=weights_path,
        mask_path=mask_path,
        output_dir=output_dir,
        output_video=args.output_video,
        annotations_json=args.annotations_json,
        coco_annotations_json=args.coco_json,
        clean_coco_annotations_json=args.clean_coco_json,
        cvat_video_xml=args.cvat_video_xml,
        labels_json=args.labels_json,
        tracker_yaml=args.tracker_yaml,
        quality_report_json=args.quality_report_json,
        quality_report_csv=args.quality_report_csv,
        start_frame=args.start_frame,
        output_fps=args.fps,
        det_conf=args.det_conf,
        track_high_conf=args.track_high_conf,
        review_conf=args.review_conf,
        adaptive_conf_step=args.adaptive_conf_step,
        conf=args.conf,
        iou=args.iou,
        class_id=args.class_id,
        allowed_class_name=args.class_name,
        use_mask=not args.no_mask,
        mask_input_frame=not args.no_mask_input,
        roi_mode=args.roi_mode,
        roi_min_cover=args.roi_min_cover,
        roi_dilate_px=args.roi_dilate_px,
        hidden_missed_frames=args.hidden_missed_frames,
        hidden_score_threshold=args.hidden_score_threshold,
        use_mask_iou=not args.no_mask_iou,
        mask_iou_max_missed=args.mask_iou_max_missed,
        mask_iou_min_area=args.mask_iou_min_area,
        max_missing_frames=args.max_missing_frames,
        match_cost_threshold=args.match_cost_threshold,
        unseen_track_cost_threshold=args.unseen_track_cost_threshold,
        lost_track_cost_threshold=args.lost_track_cost_threshold,
        lost_track_reid_appearance_threshold=(
            args.lost_track_reid_appearance_threshold
        ),
        initial_track_conf=args.initial_track_conf,
        low_conf_motion_gate=not args.no_low_conf_motion_gate,
        motion_gate_confidence=args.motion_gate_confidence,
        low_conf_max_center_jump=args.low_conf_max_center_jump,
        low_conf_max_box_jump_scale=args.low_conf_max_box_jump_scale,
        low_conf_min_iou=args.low_conf_min_iou,
        occlusion_aware_matching=not args.no_occlusion_aware_matching,
        occlusion_track_iom_threshold=args.occlusion_track_iom_threshold,
        occlusion_detection_iom_threshold=args.occlusion_detection_iom_threshold,
        occlusion_stationary_speed=args.occlusion_stationary_speed,
        occlusion_stationary_max_center_jump=(
            args.occlusion_stationary_max_center_jump
        ),
        occlusion_switch_penalty=args.occlusion_switch_penalty,
        occlusion_competitor_margin=args.occlusion_competitor_margin,
        occlusion_appearance_penalty=args.occlusion_appearance_penalty,
        occlusion_appearance_margin=args.occlusion_appearance_margin,
        occlusion_stationary_lock=not args.no_occlusion_stationary_lock,
        freeze_identity_in_occlusion=not args.learn_identity_in_occlusion,
        hold_occluded_box=not args.no_hold_occluded_box,
        occlusion_hold_max_frames=args.occlusion_hold_max_frames,
        occlusion_hold_hidden_frames=args.occlusion_hold_hidden_frames,
        identity_swap_guard=not args.no_identity_swap_guard,
        identity_swap_min_gain=args.identity_swap_min_gain,
        identity_swap_iom_threshold=args.identity_swap_iom_threshold,
        hidden_motion_model=not args.no_hidden_motion_model,
        hidden_velocity_alpha=args.hidden_velocity_alpha,
        hidden_acceleration_alpha=args.hidden_acceleration_alpha,
        hidden_stationary_speed=args.hidden_stationary_speed,
        hidden_motion_history=args.hidden_motion_history,
        hidden_min_motion_history=args.hidden_min_motion_history,
        hidden_stationary_displacement=args.hidden_stationary_displacement,
        hidden_moving_displacement=args.hidden_moving_displacement,
        hidden_motion_consistency=args.hidden_motion_consistency,
        hidden_stationary_lock_frames=args.hidden_stationary_lock_frames,
        hidden_max_motion_step_box_scale=args.hidden_max_motion_step_box_scale,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        default_behavior=args.default_behavior,
        smooth_boxes=not args.no_smooth_boxes,
        refine_boxes=not args.no_refine_boxes,
        refine_max_gap_frames=args.refine_max_gap,
        refine_size_jump_threshold=args.refine_size_jump_threshold,
        max_box_scale_change_per_frame=args.max_box_scale_change,
        max_box_scale_change_after_gap=args.max_box_scale_change_after_gap,
        high_conf_smooth_alpha=args.high_conf_smooth_alpha,
        mid_conf_smooth_alpha=args.mid_conf_smooth_alpha,
        low_conf_smooth_alpha=args.low_conf_smooth_alpha,
        max_frames=args.max_frames,
        visual_opacity=args.visual_opacity,
        show=args.show,
        display_inline=args.display_inline,
    )


def _video_paths_from_args(
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> list[Path | None]:
    if args.all_config_videos:
        return profile_video_paths(profile)
    return [None]


def print_tracking_summary(cfg: TrackingConfig, summary: TrackingSummary) -> None:
    print(f"[OK] input video: {cfg.video_path}")
    print(f"[OK] video: {summary.output_video}")
    print(f"[OK] cvat json annotations: {summary.annotations_json}")
    print(f"[OK] cvat video xml: {summary.cvat_video_xml}")
    print(f"[OK] coco annotations: {summary.coco_annotations_json}")
    print(f"[OK] clean train coco: {summary.clean_coco_annotations_json}")
    print(f"[OK] labels: {summary.labels_json}")
    print(f"[OK] quality report json: {summary.quality_report_json}")
    print(f"[OK] quality report csv: {summary.quality_report_csv}")
    print(
        "[OK] frames="
        f"{summary.frames_written}, shapes={summary.shape_count}, "
        f"hidden={summary.hidden_shape_count}, "
        f"review={summary.review_shape_count}, "
        f"start_frame={summary.start_frame}, "
        f"source_fps={summary.source_fps:.2f}, output_fps={summary.output_fps:.2f}"
    )
    print(
        "[OK] thresholds="
        f"det_conf={cfg.det_conf:.2f}, "
        f"track_high_conf={cfg.track_high_conf:.2f}, "
        f"review_conf={cfg.review_conf:.2f}, "
        f"overlap={cfg.iou:.2f}, "
        f"visual_opacity={cfg.visual_opacity:.2f}"
    )
    print(
        "[OK] low_conf_gate="
        f"enabled={cfg.low_conf_motion_gate}, "
        f"gate_conf={cfg.motion_gate_confidence:.2f}, "
        f"initial_track_conf={cfg.initial_track_conf:.2f}, "
        f"max_center_jump={cfg.low_conf_max_center_jump:.2f}"
    )
    print(
        "[OK] association="
        f"use_mask_iou={cfg.use_mask_iou}, "
        f"mask_iou_max_missed={cfg.mask_iou_max_missed}, "
        f"mask_iou_min_area={cfg.mask_iou_min_area}, "
        f"bbox_fallback=True"
    )
    print(
        "[OK] occlusion_matching="
        f"enabled={cfg.occlusion_aware_matching}, "
        f"track_iom={cfg.occlusion_track_iom_threshold:.2f}, "
        f"detection_iom={cfg.occlusion_detection_iom_threshold:.2f}, "
        f"stationary_jump={cfg.occlusion_stationary_max_center_jump:.3f}, "
        f"stationary_lock={cfg.occlusion_stationary_lock}, "
        f"freeze_identity={cfg.freeze_identity_in_occlusion}, "
        f"hold_box={cfg.hold_occluded_box}, "
        f"hold_hidden_frames={cfg.occlusion_hold_hidden_frames}"
    )
    print(
        "[OK] identity_swap_guard="
        f"enabled={cfg.identity_swap_guard}, "
        f"min_gain={cfg.identity_swap_min_gain:.3f}, "
        f"iom={cfg.identity_swap_iom_threshold:.2f}"
    )
    print(
        "[OK] hidden_motion="
        f"enabled={cfg.hidden_motion_model}, "
        f"stationary_speed={cfg.hidden_stationary_speed:.3f}, "
        f"history={cfg.hidden_motion_history}, "
        f"min_history={cfg.hidden_min_motion_history}, "
        f"moving_disp={cfg.hidden_moving_displacement:.3f}, "
        f"lock_frames={cfg.hidden_stationary_lock_frames}, "
        f"max_step_scale={cfg.hidden_max_motion_step_box_scale:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_config_videos and any(
        path is not None
        for path in (
            args.output_video,
            args.annotations_json,
            args.coco_json,
            args.clean_coco_json,
            args.cvat_video_xml,
            args.labels_json,
            args.tracker_yaml,
            args.quality_report_json,
            args.quality_report_csv,
        )
    ):
        raise ValueError(
            "Do not use single-output file arguments with --all-config-videos."
        )

    profile = _profile_from_args(args)
    summaries: list[TrackingSummary] = []
    for video_path in _video_paths_from_args(args, profile):
        cfg = _tracking_config_from_args(args, profile, video_path)
        summary = run_tracking(cfg)
        summaries.append(summary)
        print_tracking_summary(cfg, summary)
        if args.display_inline:
            display_tracked_video(summary.output_video)

    if len(summaries) > 1:
        print(f"[OK] processed videos: {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
