from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class YoloPigDetector:
    def __init__(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights do not exist: {weights_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for detection. Install project optional dependency 'tracking'."
            ) from exc
        self.model = YOLO(str(weights_path))

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        results = self.model.predict(frame, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                detections.append(
                    Detection(
                        frame_index=frame_index,
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=confidence,
                    )
                )
        return detections
