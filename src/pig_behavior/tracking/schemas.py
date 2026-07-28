"""Data structures for fixed-ID pig tracking state."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.constants import TRACKING_TELEMETRY_KEYS


@dataclass(slots=True)
class Detection:
    """One YOLO/ByteTrack detection after ROI filtering."""

    box: np.ndarray
    score: float
    raw_id: int | None
    class_id: int | None
    hist: np.ndarray
    mask: np.ndarray | None = None
    core_hist: np.ndarray | None = None


@dataclass(slots=True)
class OcclusionContext:
    """Ambiguous track/detection relationships in crowded overlap zones."""

    predicted_boxes: dict[int, np.ndarray]
    occluded_track_ids: set[int]
    detection_competitors: dict[int, set[int]]
    active_detection_owners: dict[int, set[int]]
    appearance_costs: dict[tuple[int, int], float]


@dataclass(slots=True)
class ConflictGroup:
    """Local track/detection component that may require occlusion handling."""

    track_ids: set[int]
    detection_indices: set[int]


@dataclass(slots=True)
class HardSceneDecision:
    """Rule-based classification for a local conflict group."""

    state: str
    is_hard_occlusion: bool
    is_merged: bool
    score: float
    has_detection_deficit: bool
    has_oversized_detection: bool


@dataclass(slots=True)
class TrackingRuntimeState:
    """State carried across frames for local FSM and telemetry."""

    group_states: dict[tuple[int, ...], str] = field(default_factory=dict)
    group_hard_frames: dict[tuple[int, ...], int] = field(default_factory=dict)
    group_recovery_remaining: dict[tuple[int, ...], int] = field(default_factory=dict)
    current_recovery_track_ids: set[int] = field(default_factory=set)
    telemetry: dict[str, int | float | str] = field(
        default_factory=lambda: {key: 0 for key in TRACKING_TELEMETRY_KEYS}
    )
    timing_samples_seconds: dict[str, list[float]] = field(default_factory=dict)
    association_debug_events: list[dict[str, object]] = field(default_factory=list)
    reentry_unowned_raw_quarantine: dict[int, int] = field(default_factory=dict)
    reentry_unowned_raw_episode_history: dict[tuple[int, int, int], list[int]] = field(
        default_factory=dict
    )
    occlusion_reid_bad_match_hold_keys: set[tuple[int, int | None]] = field(
        default_factory=set
    )


@dataclass(slots=True)
class FixedTrack:
    """Stable ID state. Each video frame will emit one box per FixedTrack."""

    fixed_id: int
    last_box: np.ndarray
    reliable_box: np.ndarray | None = None
    hist_bank: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=80))
    core_hist_bank: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=80)
    )
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
    is_area_occluded: bool = False
    area_occlusion_frames: int = 0
    last_merged_split: bool = False
    hard_occlusion_frames: int = 0
    hard_occlusion_recovery_frames: int = 0
    ever_detected: bool = False
    state: str = "MISSING"
    state_reason: str = "initialized"
    occlusion_count: int = 0

    @property
    def missed_count(self) -> int:
        return self.missed

    @missed_count.setter
    def missed_count(self, value: int) -> None:
        self.missed = value

    def mean_hist(self) -> np.ndarray | None:
        if not self.hist_bank:
            return None
        return np.mean(np.stack(tuple(self.hist_bank), axis=0), axis=0)

    def mean_core_hist(self) -> np.ndarray | None:
        if not self.core_hist_bank:
            return None
        return np.mean(np.stack(tuple(self.core_hist_bank), axis=0), axis=0)

    def top_raw_id(self) -> int | None:
        if not self.raw_id_counts:
            return None
        return self.raw_id_counts.most_common(1)[0][0]

    def get_state(self) -> str:
        return self.state

    def predicted_box(self, width: int, height: int) -> np.ndarray:
        damping = 0.85 ** min(self.missed, 12)
        if self.missed > 90:
            damping = 0.0
        dx, dy = self.velocity_xy * damping
        box = self.last_box.copy()
        box[[0, 2]] += dx
        box[[1, 3]] += dy
        return _clip_box(box, width, height)

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
        new_center = np.array(_bbox_center(new_box), dtype=np.float32)
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
            return _clip_box(reference.copy(), width, height)

        if self.motion_state != "moving":
            reference = (
                self.reliable_box if self.reliable_box is not None else self.last_box
            )
            return _clip_box(reference.copy(), width, height)

        speed = self.reliable_speed_norm(width, height)
        if speed <= cfg.hidden_stationary_speed:
            reference = (
                self.reliable_box if self.reliable_box is not None else self.last_box
            )
            return _clip_box(reference.copy(), width, height)

        if self.occlusion_hold_frames == 0 and self.reliable_box is not None:
            reference = self.reliable_box.copy()
        else:
            reference = self.last_box.copy()
        displacement = self.reliable_velocity_xy + 0.5 * self.reliable_acceleration_xy
        displacement *= 0.90 ** min(self.missed + self.occlusion_hold_frames, 12)

        box_w, box_h = _bbox_size(reference)
        max_step = cfg.hidden_max_motion_step_box_scale * math.hypot(box_w, box_h)
        step_norm = float(np.linalg.norm(displacement))
        if max_step > 0.0 and step_norm > max_step:
            displacement *= max_step / max(step_norm, 1e-6)

        predicted = reference.copy()
        predicted[[0, 2]] += displacement[0]
        predicted[[1, 3]] += displacement[1]
        return _clip_box(predicted, width, height)

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
            and _detection_needs_motion_gate(det, cfg)
            and self.reliable_box is not None
        ):
            previous_box = self.reliable_box

        previous_center = _bbox_center(previous_box)
        new_box = _clip_box(det.box, width, height)
        if self.ever_detected and cfg.smooth_boxes:
            new_box = _smooth_detected_box(
                previous_box,
                new_box,
                det.score,
                self.missed,
                width,
                height,
                cfg,
            )
        new_center = _bbox_center(new_box)
        self.velocity_xy = np.array(
            [new_center[0] - previous_center[0], new_center[1] - previous_center[1]],
            dtype=np.float32,
        )
        self.last_box = new_box
        self.last_score = float(det.score)
        self.last_source = "detected"
        self.last_ambiguous = bool(ambiguous)
        self.is_area_occluded = False
        self.area_occlusion_frames = 0
        self.last_merged_split = False
        self.last_mask = det.mask.copy() if det.mask is not None else None
        self.missed = 0
        self.missed_count = 0
        self.hits += 1
        self.ever_detected = True

        if ambiguous or self.is_area_occluded or self.last_merged_split:
            self.state = "OCCLUDED"
            self.state_reason = (
                "detected_ambiguous" if ambiguous else
                "area_occlusion" if self.is_area_occluded else
                "merged_split"
            )
            self.occlusion_count += 1
        else:
            self.state = "VISIBLE"
            self.state_reason = (
                "detected_high_conf"
                if det.score >= cfg.track_high_conf
                else "detected_low_conf"
            )
            self.occlusion_count = 0

        if learn_identity:
            self.hist_bank.append(det.hist)
        if learn_identity and det.core_hist is not None:
            self.core_hist_bank.append(det.core_hist)
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
        cfg: TrackingConfig | None = None,
        is_skip_frame: bool = False,
    ) -> None:
        previous_center = _bbox_center(self.last_box)
        new_box = _clip_box(box, width, height)
        new_center = _bbox_center(new_box)
        self.velocity_xy = 0.5 * self.velocity_xy + 0.5 * np.array(
            [new_center[0] - previous_center[0], new_center[1] - previous_center[1]],
            dtype=np.float32,
        )
        self.last_box = new_box
        if not is_skip_frame:
            self.last_score = max(0.05, self.last_score * 0.92)
            self.last_source = "occlusion_hold" if hold else "predicted"
            self.last_ambiguous = bool(ambiguous)
            self.last_merged_split = False
            self.missed += 1
            self.missed_count = self.missed
            if hold:
                self.occlusion_hold_frames += 1
            else:
                self.occlusion_hold_frames = 0
                self.is_area_occluded = False
                self.area_occlusion_frames = 0

            if hold or self.is_area_occluded:
                self.state = "OCCLUDED"
                self.state_reason = "occlusion_hold"
                self.occlusion_count += 1
            else:
                max_missing = cfg.max_missing_frames if cfg is not None else 30
                if self.missed_count > max_missing:
                    self.state = "LOST"
                    self.state_reason = "max_missing_exceeded"
                else:
                    self.state = "MISSING"
                    self.state_reason = "predicted"
                self.occlusion_count = 0
        else:
            self.state_reason = "prediction_only"


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
    telemetry: dict[str, int | float | str]


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float32).copy()
    out[0] = max(0.0, min(float(width - 1), float(out[0])))
    out[1] = max(0.0, min(float(height - 1), float(out[1])))
    out[2] = max(out[0] + 1.0, min(float(width), float(out[2])))
    out[3] = max(out[1] + 1.0, min(float(height), float(out[3])))
    return out


def _bbox_area(box: np.ndarray) -> float:
    return max(1.0, float(box[2] - box[0])) * max(1.0, float(box[3] - box[1]))


def _bbox_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def _bbox_size(box: np.ndarray) -> tuple[float, float]:
    return max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1]))


def _smooth_alpha_for_score(score: float, cfg: TrackingConfig) -> float:
    if score >= cfg.review_conf:
        return cfg.high_conf_smooth_alpha
    if score >= cfg.track_high_conf:
        return cfg.mid_conf_smooth_alpha
    return cfg.low_conf_smooth_alpha


def _smooth_detected_box(
    previous_box: np.ndarray,
    detected_box: np.ndarray,
    score: float,
    missed_frames: int,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    """Limit sudden box-size jumps while keeping motion responsive."""
    previous_box = _clip_box(previous_box, width, height)
    detected_box = _clip_box(detected_box, width, height)
    prev_cx, prev_cy = _bbox_center(previous_box)
    det_cx, det_cy = _bbox_center(detected_box)
    prev_w, prev_h = _bbox_size(previous_box)
    det_w, det_h = _bbox_size(detected_box)

    max_scale_change = (
        cfg.max_box_scale_change_after_gap
        if missed_frames > 0
        else cfg.max_box_scale_change_per_frame
    )
    min_scale = max(0.05, 1.0 - max_scale_change)
    max_scale = 1.0 + max_scale_change
    limited_w = float(np.clip(det_w, prev_w * min_scale, prev_w * max_scale))
    limited_h = float(np.clip(det_h, prev_h * min_scale, prev_h * max_scale))

    alpha = _smooth_alpha_for_score(score, cfg)
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
    return _clip_box(smoothed, width, height)


def _detection_needs_motion_gate(det: Detection, cfg: TrackingConfig) -> bool:
    return cfg.low_conf_motion_gate and det.score < cfg.motion_gate_confidence


__all__ = [
    "ConflictGroup",
    "Detection",
    "FixedTrack",
    "HardSceneDecision",
    "OcclusionContext",
    "TrackingRuntimeState",
    "TrackingSummary",
]
