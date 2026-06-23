"""Real-time video detection, tracking, and sequence behavior statistics."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from pig_behavior.config import (
    BEHAVIOR_CLASSIFIER_WEIGHTS,
    BEHAVIOR_SEQUENCE_OFFSETS,
    BEHAVIOR_SEQUENCE_STRIDE_FRAMES,
    COCO_ANNOTATIONS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_VIDEO_PATH,
)
from pig_behavior.models.behavior_sequence import (
    BehaviorFrameSample,
    BehaviorSequenceClassifier,
)


@dataclass(slots=True)
class TrackingConfig:
    """Runtime settings for a video tracking session."""

    detector_model_path: Path = DEFAULT_DETECTOR_MODEL
    behavior_model_path: Path = BEHAVIOR_CLASSIFIER_WEIGHTS
    video_path: Path = DEFAULT_VIDEO_PATH
    confidence: float = 0.25
    frame_stride: int = 1
    behavior_stride_frames: int = BEHAVIOR_SEQUENCE_STRIDE_FRAMES
    realtime: bool = True


class VideoTrackingSession:
    """Background video processor with MJPEG streaming state."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._lock = Lock()
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._error: str | None = None
        self._running = False
        self._reset_stats()

    @property
    def running(self) -> bool:
        """Return whether processing is active."""
        with self._lock:
            return self._running

    def start(self, config: TrackingConfig | None = None) -> None:
        """Start processing in the background."""
        with self._lock:
            if self._running:
                return
            if config is not None:
                self.config = config
            self._validate_config()
            self._reset_stats()
            self._error = None
            self._stop_event.clear()
            self._running = True
            self._worker = Thread(target=self._run, daemon=True)
            self._worker.start()

    def stop(self) -> None:
        """Request processing to stop."""
        self._stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable status snapshot."""
        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "detector_model_path": str(self.config.detector_model_path),
                "behavior_model_path": str(self.config.behavior_model_path),
                "model_path": str(self.config.detector_model_path),
                "video_path": str(self.config.video_path),
                "confidence": self.config.confidence,
                "frame_stride": self.config.frame_stride,
                "behavior_stride_frames": self.config.behavior_stride_frames,
                "sequence_offsets": list(BEHAVIOR_SEQUENCE_OFFSETS),
                "realtime": self.config.realtime,
                "frame_index": self._frame_index,
                "frames_processed": self._frames_processed,
                "total_frames": self._total_frames,
                "source_fps": round(self._source_fps, 2),
                "processing_fps": round(self._processing_fps, 2),
                "video_time_sec": round(self._video_time_sec, 2),
                "track_count": len(self._track_ids_seen),
                "current_counts": dict(self._current_counts),
                "cumulative_counts": dict(self._cumulative_counts),
                "behavior_seconds": {
                    label: round(seconds, 2)
                    for label, seconds in self._behavior_seconds.items()
                },
                "top_tracks": self._top_tracks_unlocked(),
                "history": list(self._history),
            }

    def frame_stream(self) -> Iterator[bytes]:
        """Yield the latest JPEG frame as an MJPEG stream."""
        while True:
            with self._lock:
                frame = self._latest_jpeg
                running = self._running
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            elif not running:
                time.sleep(0.25)
            time.sleep(0.04)

    def _validate_config(self) -> None:
        if not self.config.detector_model_path.exists():
            raise FileNotFoundError(
                f"Detector model not found: {self.config.detector_model_path}"
            )
        if not self.config.behavior_model_path.exists():
            raise FileNotFoundError(
                f"Behavior classifier not found: {self.config.behavior_model_path}"
            )
        if not self.config.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.config.video_path}")
        if self.config.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        if self.config.behavior_stride_frames < 1:
            raise ValueError("behavior_stride_frames must be at least 1.")
        if self.config.confidence <= 0 or self.config.confidence >= 1:
            raise ValueError("confidence must be between 0 and 1.")

    def _reset_stats(self) -> None:
        self._frame_index = 0
        self._frames_processed = 0
        self._total_frames = 0
        self._source_fps = 0.0
        self._processing_fps = 0.0
        self._video_time_sec = 0.0
        self._current_counts: Counter[str] = Counter()
        self._cumulative_counts: Counter[str] = Counter()
        self._behavior_seconds: defaultdict[str, float] = defaultdict(float)
        self._track_ids_seen: set[int] = set()
        self._track_history: defaultdict[int, dict[int, dict[str, Any]]]
        self._track_history = defaultdict(dict)
        self._frame_detections: dict[int, list[dict[str, Any]]] = {}
        self._track_latest_behavior: dict[int, tuple[str, float]] = {}
        self._track_behavior_counts: defaultdict[int, Counter[str]]
        self._track_behavior_counts = defaultdict(Counter)
        self._history: deque[dict[str, Any]] = deque(maxlen=180)
        self._last_history_second = -1
        self._started_at = 0.0
        self._roi_boxes: dict[str, list[tuple[float, float, float, float]]] = {}

    def _run(self) -> None:
        try:
            self._process_video()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False

    def _process_video(self) -> None:
        try:
            import cv2
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Video tracking requires ultralytics and opencv. "
                "Install with `pip install -e .[pt]`."
            ) from exc

        detector = YOLO(str(self.config.detector_model_path))
        behavior_classifier = BehaviorSequenceClassifier(
            self.config.behavior_model_path
        )
        behavior_classifier.load()

        capture = cv2.VideoCapture(str(self.config.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.config.video_path}")

        self._started_at = time.perf_counter()
        self._source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        self._total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_delay = 1.0 / max(self._source_fps, 1.0)

        try:
            while not self._stop_event.is_set():
                loop_started_at = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break

                self._frame_index += 1
                if self._frame_index % self.config.frame_stride != 0:
                    continue

                if not self._roi_boxes:
                    height, width = frame.shape[:2]
                    self._roi_boxes = _load_roi_boxes(COCO_ANNOTATIONS, width, height)

                results = detector.track(
                    source=frame,
                    persist=True,
                    conf=self.config.confidence,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                result = results[0]
                detections = self._parse_detections(result, frame)
                self._store_track_samples(detections)
                self._update_delayed_behavior_predictions(behavior_classifier)
                self._attach_latest_behavior(detections)

                annotated = result.plot()
                self._draw_behavior_labels(annotated, detections)
                self._update_stats(detections)
                self._store_frame(annotated)

                if self.config.realtime:
                    elapsed = time.perf_counter() - loop_started_at
                    time.sleep(max(0.0, frame_delay - elapsed))
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _parse_detections(self, result: Any, frame: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        import cv2

        names = _names_dict(getattr(result, "names", {}))
        class_ids = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        bboxes = boxes.xyxy.detach().cpu().tolist()
        track_ids = (
            boxes.id.detach().cpu().tolist()
            if getattr(boxes, "id", None) is not None
            else [None] * len(class_ids)
        )

        height, width = frame.shape[:2]
        detections = []
        for class_id, confidence, track_id, bbox in zip(
            class_ids,
            confidences,
            track_ids,
            bboxes,
            strict=True,
        ):
            clipped = _clip_bbox(bbox, width, height)
            x1, y1, x2, y2 = [int(round(value)) for value in clipped]
            crop_bgr = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            cx = (clipped[0] + clipped[2]) / 2.0
            cy = (clipped[1] + clipped[3]) / 2.0
            detections.append(
                {
                    "frame_index": self._frame_index,
                    "detector_label": names.get(
                        int(class_id),
                        f"class_{int(class_id)}",
                    ),
                    "detector_confidence": float(confidence),
                    "track_id": int(track_id) if track_id is not None else None,
                    "bbox_xyxy": [float(value) for value in clipped],
                    "center_xy": (cx, cy),
                    "crop_rgb": crop_rgb,
                    "frame_size": (width, height),
                    "behavior_label": "pending",
                    "behavior_confidence": 0.0,
                }
            )
        return detections

    def _store_track_samples(self, detections: list[dict[str, Any]]) -> None:
        frame_items = [
            {
                "track_id": detection["track_id"],
                "center_xy": detection["center_xy"],
            }
            for detection in detections
        ]
        self._frame_detections[self._frame_index] = frame_items

        for detection in detections:
            track_id = detection["track_id"]
            if track_id is None:
                continue
            self._track_ids_seen.add(track_id)
            self._track_history[track_id][self._frame_index] = detection

        keep_after = self._frame_index - max(
            240,
            self.config.behavior_stride_frames * 24,
        )
        for history in self._track_history.values():
            for frame_index in [idx for idx in history if idx < keep_after]:
                history.pop(frame_index, None)
        for frame_index in [idx for idx in self._frame_detections if idx < keep_after]:
            self._frame_detections.pop(frame_index, None)

    def _update_delayed_behavior_predictions(
        self,
        classifier: BehaviorSequenceClassifier,
    ) -> None:
        future_delay = (
            max(BEHAVIOR_SEQUENCE_OFFSETS)
            * self.config.behavior_stride_frames
        )
        center_frame = self._frame_index - future_delay
        if center_frame < 0:
            return

        for track_id in list(self._track_history):
            samples = self._sequence_samples_for_track(track_id, center_frame)
            if samples is None:
                continue
            prediction = classifier.predict(samples)
            self._track_latest_behavior[track_id] = (
                prediction.label,
                prediction.confidence,
            )
            self._track_behavior_counts[track_id].update([prediction.label])

    def _sequence_samples_for_track(
        self,
        track_id: int,
        center_frame: int,
    ) -> list[BehaviorFrameSample] | None:
        history = self._track_history.get(track_id)
        if not history:
            return None

        selected = []
        for offset in BEHAVIOR_SEQUENCE_OFFSETS:
            target = center_frame + offset * self.config.behavior_stride_frames
            nearest = _nearest_key(history, target)
            if nearest is None:
                return None
            selected.append(history[nearest])

        features = self._sequence_features(selected)
        return [
            BehaviorFrameSample(
                crop_rgb=detection["crop_rgb"],
                features=feature_row,
                frame_index=detection["frame_index"],
                bbox_xyxy=detection["bbox_xyxy"],
            )
            for detection, feature_row in zip(selected, features, strict=True)
        ]

    def _sequence_features(self, detections: list[dict[str, Any]]) -> list[list[float]]:
        features: list[list[float]] = []
        previous_center: tuple[float, float] | None = None
        for detection in detections:
            width, height = detection["frame_size"]
            diag = math.sqrt(width**2 + height**2)
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            cx, cy = detection["center_xy"]
            bw = x2 - x1
            bh = y2 - y1
            speed = 0.0
            if previous_center is not None:
                speed = math.dist((cx, cy), previous_center) / max(diag, 1e-9)
            previous_center = (cx, cy)

            min_dist_other, num_close_other = self._neighbor_features(
                detection["frame_index"],
                detection["track_id"],
                cx,
                cy,
                diag,
            )
            in_feeder, in_drinker, in_toy = self._roi_flags(detection["bbox_xyxy"])
            features.append(
                [
                    cx / max(width, 1),
                    cy / max(height, 1),
                    bw / max(width, 1),
                    bh / max(height, 1),
                    speed,
                    min_dist_other,
                    num_close_other,
                    in_feeder,
                    in_drinker,
                    in_toy,
                ]
            )
        return features

    def _neighbor_features(
        self,
        frame_index: int,
        track_id: int | None,
        cx: float,
        cy: float,
        diag: float,
    ) -> tuple[float, float]:
        dists = []
        for item in self._frame_detections.get(frame_index, []):
            if item["track_id"] == track_id:
                continue
            ox, oy = item["center_xy"]
            dists.append(math.dist((cx, cy), (ox, oy)) / max(diag, 1e-9))
        if not dists:
            return 0.0, 0.0
        return min(dists), float(sum(distance < 0.12 for distance in dists))

    def _roi_flags(self, bbox: list[float]) -> tuple[float, float, float]:
        flags = []
        for name in ("feeder", "drinker", "toy"):
            inside = any(
                _rect_intersection_area(tuple(bbox), roi_box) > 0
                for roi_box in self._roi_boxes.get(name, [])
            )
            flags.append(1.0 if inside else 0.0)
        return flags[0], flags[1], flags[2]

    def _attach_latest_behavior(self, detections: list[dict[str, Any]]) -> None:
        for detection in detections:
            track_id = detection["track_id"]
            if track_id is None:
                detection["behavior_label"] = "unknown"
                detection["behavior_confidence"] = 0.0
                continue
            label, confidence = self._track_latest_behavior.get(
                track_id,
                ("pending", 0.0),
            )
            detection["behavior_label"] = label
            detection["behavior_confidence"] = confidence

    def _update_stats(self, detections: list[dict[str, Any]]) -> None:
        valid_labels = [
            detection["behavior_label"]
            for detection in detections
            if detection["behavior_label"] not in {"pending", "unknown"}
        ]
        current_counts = Counter(valid_labels)
        video_time_sec = self._frame_index / max(self._source_fps, 1.0)
        seconds_per_processed_frame = (
            self.config.frame_stride / max(self._source_fps, 1.0)
        )

        with self._lock:
            self._frames_processed += 1
            self._video_time_sec = video_time_sec
            self._current_counts = current_counts
            self._cumulative_counts.update(current_counts)
            for label, count in current_counts.items():
                self._behavior_seconds[label] += count * seconds_per_processed_frame

            elapsed = max(time.perf_counter() - self._started_at, 1e-6)
            self._processing_fps = self._frames_processed / elapsed
            current_second = int(video_time_sec)
            if current_second != self._last_history_second:
                self._last_history_second = current_second
                self._history.append(
                    {
                        "time_sec": round(video_time_sec, 2),
                        "counts": dict(current_counts),
                    }
                )

    def _store_frame(self, frame: Any) -> None:
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

    def _draw_behavior_labels(
        self,
        frame: Any,
        detections: list[dict[str, Any]],
    ) -> None:
        import cv2

        for detection in detections:
            x1, y1, _x2, _y2 = [
                int(round(value)) for value in detection["bbox_xyxy"]
            ]
            track_id = detection["track_id"]
            prefix = f"ID {track_id} " if track_id is not None else ""
            label = (
                f"{prefix}{detection['behavior_label']} "
                f"{detection['behavior_confidence']:.2f}"
            )
            cv2.putText(
                frame,
                label,
                (max(x1, 0), max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (34, 197, 94),
                2,
                cv2.LINE_AA,
            )

    def _top_tracks_unlocked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for track_id, counts in self._track_behavior_counts.items():
            if not counts:
                continue
            label, count = counts.most_common(1)[0]
            rows.append(
                {
                    "track_id": track_id,
                    "dominant_behavior": label,
                    "observations": count,
                    "counts": dict(counts),
                }
            )
        rows.sort(key=lambda item: item["observations"], reverse=True)
        return rows[:20]


def _clip_bbox(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(x1 + 1.0, min(x2, float(width)))
    y2 = max(y1 + 1.0, min(y2, float(height)))
    return x1, y1, x2, y2


def _nearest_key(items: dict[int, Any], target: int) -> int | None:
    if not items:
        return None
    return min(items, key=lambda key: abs(key - target))


def _load_roi_boxes(
    coco_path: Path,
    image_width: int,
    image_height: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    if not coco_path.exists():
        return {}
    with coco_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    base_image = coco.get("images", [{}])[0]
    base_width = float(base_image.get("width") or image_width)
    base_height = float(base_image.get("height") or image_height)
    scale_x = image_width / max(base_width, 1.0)
    scale_y = image_height / max(base_height, 1.0)
    categories = {
        category["id"]: category["name"]
        for category in coco.get("categories", [])
    }

    boxes: defaultdict[str, list[tuple[float, float, float, float]]]
    boxes = defaultdict(list)
    for annotation in coco.get("annotations", []):
        name = categories.get(annotation.get("category_id"))
        if name not in {"feeder", "drinker", "toy"}:
            continue
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, width, height = [float(value) for value in bbox]
        margin = 8.0
        boxes[name].append(
            (
                x * scale_x - margin,
                y * scale_y - margin,
                (x + width) * scale_x + margin,
                (y + height) * scale_y + margin,
            )
        )
    return dict(boxes)


def _rect_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}
