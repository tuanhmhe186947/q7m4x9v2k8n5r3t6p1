from __future__ import annotations

from pathlib import Path

import cv2


def count_video_frames(video_path: str | Path) -> int | None:
    path = Path(video_path)
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count >= 0 else None
    finally:
        cap.release()


class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise OSError(f"Could not open video: {self.video_path}")

    def read(self, frame_index: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

