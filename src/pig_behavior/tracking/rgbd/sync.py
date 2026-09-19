# ruff: noqa
"""Synchronise colour and depth video streams for RGB-D tracking.

# ruff: noqa

Supports three alignment modes:

1. **Timestamp file** (``times.txt``): Two-column file mapping colour and
   depth timestamps.  Each colour frame is aligned to the nearest depth
   frame by time.
2. **Index mapping file**: If ``times.txt`` has a single integer per line it
   is treated as a direct colour→depth index map.
3. **1-to-1 fallback**: When no ``times.txt`` is given *and* the frame counts
   match, depth frame *i* is paired with colour frame *i*.  A warning is
   logged so the user is aware of the implicit alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Diagnostic counters emitted after synchroniser construction."""

    color_frame_count: int = 0
    depth_frame_count: int = 0
    timestamp_lines: int = 0
    mean_delta_ms: float = 0.0
    max_delta_ms: float = 0.0
    sync_mode: str = "unknown"


class RGBDFrameSynchronizer:
    """Map colour frame indices to their nearest depth frame indices.

    Parameters
    ----------
    color_video_path:
        Path to the colour ``.mp4`` file.
    depth_video_path:
        Path to the depth ``.mp4`` file.
    times_path:
        Optional path to ``times.txt``.  If *None* the synchroniser falls
        back to 1-to-1 index mapping.
    """

    def __init__(
        self,
        color_video_path: Path,
        depth_video_path: Path,
        times_path: Path | None = None,
    ) -> None:
        import cv2

        self._color_cap = cv2.VideoCapture(str(color_video_path))
        self._depth_cap = cv2.VideoCapture(str(depth_video_path))

        if not self._color_cap.isOpened():
            raise RuntimeError(f"Cannot open colour video: {color_video_path}")
        if not self._depth_cap.isOpened():
            raise RuntimeError(f"Cannot open depth video: {depth_video_path}")

        self._color_count = int(self._color_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._depth_count = int(self._depth_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._mapping: np.ndarray | None = None
        self._color_pos = 0
        self._depth_pos = 0
        self._stats = SyncStats(
            color_frame_count=self._color_count,
            depth_frame_count=self._depth_count,
        )

        if times_path is not None and times_path.exists():
            self._build_from_times_file(times_path)
        else:
            self._build_identity_mapping(times_path)

        logger.info(
            "RGBDFrameSynchronizer ready: color=%d depth=%d mode=%s "
            "mean_delta=%.2fms max_delta=%.2fms",
            self._stats.color_frame_count,
            self._stats.depth_frame_count,
            self._stats.sync_mode,
            self._stats.mean_delta_ms,
            self._stats.max_delta_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SyncStats:
        """Return diagnostic counters."""
        return self._stats

    @property
    def color_frame_count(self) -> int:
        return self._color_count

    @property
    def depth_frame_count(self) -> int:
        return self._depth_count

    def get_depth_index(self, color_index: int) -> int:
        """Return the depth frame index nearest to *color_index*."""
        if self._mapping is not None and 0 <= color_index < len(self._mapping):
            return int(self._mapping[color_index])
        return min(color_index, max(0, self._depth_count - 1))

    def read_synced(
        self,
        color_index: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
        """Read the colour frame and its nearest depth frame.

        Returns ``(color_frame, depth_frame, depth_index)``.
        Any element may be *None* if the seek or read fails.
        """

        color_frame = self._seek_and_read(self._color_cap, color_index, "_color_pos")
        depth_idx = self.get_depth_index(color_index)
        depth_frame = self._seek_and_read(self._depth_cap, depth_idx, "_depth_pos")
        return color_frame, depth_frame, depth_idx

    def release(self) -> None:
        """Release both video captures."""
        self._color_cap.release()
        self._depth_cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seek_and_read(
        self,
        cap: object,
        index: int,
        pos_attr: str,
    ) -> np.ndarray | None:
        import cv2

        current_pos = getattr(self, pos_attr)
        if index != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # type: ignore[union-attr]
            setattr(self, pos_attr, index)

        ok, frame = cap.read()  # type: ignore[union-attr]
        if ok:
            setattr(self, pos_attr, index + 1)
        return frame if ok else None

    def _build_from_times_file(self, times_path: Path) -> None:
        """Parse ``times.txt`` and build the colour→depth mapping."""
        lines = times_path.read_text(encoding="utf-8").strip().splitlines()
        self._stats.timestamp_lines = len(lines)

        # Heuristic: single-column integers → direct index map
        try:
            values = [line.strip().split() for line in lines if line.strip()]
            if all(len(v) == 1 for v in values):
                mapping = np.array([int(v[0]) for v in values], dtype=np.int64)
                mapping = np.clip(mapping, 0, max(0, self._depth_count - 1))
                self._mapping = mapping
                self._stats.sync_mode = "index_map"
                logger.info(
                    "times.txt parsed as direct index map (%d entries)", len(mapping)
                )
                return
        except ValueError:
            pass

        # Two-column timestamps: [color_ts, depth_ts] or interleaved blocks
        try:
            color_ts: list[float] = []
            depth_ts: list[float] = []
            for parts in values:
                if len(parts) >= 2:
                    color_ts.append(float(parts[0]))
                    depth_ts.append(float(parts[1]))
            if color_ts and depth_ts:
                self._build_from_timestamps(
                    np.array(color_ts), np.array(depth_ts)
                )
                self._stats.sync_mode = "timestamp_pairs"
                return
        except (ValueError, IndexError):
            pass

        # Fallback: single-column float timestamps (assume one list per stream)
        try:
            all_ts = np.array([float(v[0]) for v in values])
            half = len(all_ts) // 2
            if half > 0:
                self._build_from_timestamps(all_ts[:half], all_ts[half:])
                self._stats.sync_mode = "timestamp_split"
                return
        except (ValueError, IndexError):
            pass

        logger.warning(
            "Could not parse times.txt at %s — falling back to 1-to-1 mapping",
            times_path,
        )
        self._build_identity_mapping(None)

    def _build_from_timestamps(
        self,
        color_ts: np.ndarray,
        depth_ts: np.ndarray,
    ) -> None:
        """Nearest-neighbour mapping from colour timestamps to depth timestamps."""
        mapping = np.searchsorted(depth_ts, color_ts, side="left")
        # Snap to the closer of the two neighbours
        for i in range(len(mapping)):
            idx = mapping[i]
            if idx >= len(depth_ts):
                idx = len(depth_ts) - 1
            elif idx > 0:
                d_left = abs(color_ts[i] - depth_ts[idx - 1])
                d_right = abs(color_ts[i] - depth_ts[idx])
                if d_left < d_right:
                    idx = idx - 1
            mapping[i] = idx

        deltas_ms = np.abs(color_ts - depth_ts[mapping]) * 1000.0
        self._stats.mean_delta_ms = float(np.mean(deltas_ms))
        self._stats.max_delta_ms = float(np.max(deltas_ms))
        self._mapping = np.clip(mapping, 0, max(0, self._depth_count - 1))
        logger.info(
            "Timestamp sync built: %d entries, mean_delta=%.2fms max_delta=%.2fms",
            len(mapping),
            self._stats.mean_delta_ms,
            self._stats.max_delta_ms,
        )

    def _build_identity_mapping(self, times_path: Path | None) -> None:
        """1-to-1 fallback when no timestamp file is available."""
        if times_path is not None:
            logger.warning(
                "times.txt path provided (%s) but file does not exist — "
                "using 1-to-1 index mapping",
                times_path,
            )
        if self._color_count != self._depth_count and self._depth_count > 0:
            logger.warning(
                "Frame counts differ (color=%d, depth=%d) — "
                "depth indices will be clamped",
                self._color_count,
                self._depth_count,
            )
        self._mapping = None
        self._stats.sync_mode = "identity"


__all__ = [
    "RGBDFrameSynchronizer",
    "SyncStats",
]
