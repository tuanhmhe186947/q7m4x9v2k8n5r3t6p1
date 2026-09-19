from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class RuntimeReporter:
    def __init__(self, output_root: Path, log_file: Path | None = None):
        self.output_root = output_root
        self.log_file = log_file
        self.stage_timings: dict[str, float] = {}
        self.started_at = time.perf_counter()
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        elapsed = time.perf_counter() - self.started_at
        line = f"[{elapsed:9.2f}s] {message}"
        print(line, flush=True)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.log(f"START stage={name}")
        started = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            self.stage_timings[f"{name}_sec"] = duration
            self.log(f"END stage={name} duration_sec={duration:.2f}")

    def total_sec(self) -> float:
        return time.perf_counter() - self.started_at

    def write_timing_report(self) -> Path:
        report = dict(self.stage_timings)
        report["total_sec"] = self.total_sec()
        path = self.output_root / "timing_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.log(f"WROTE {path}")
        return path


def write_progress_state(
    output_root: Path,
    processed_tracklets: int,
    current_group_id: object,
    current_video: object,
    stage: str,
    elapsed_sec: float,
) -> Path:
    path = output_root / "progress_state.json"
    state = {
        "processed_tracklet_count": processed_tracklets,
        "current_group_id": "" if current_group_id is None else str(current_group_id),
        "current_video": "" if current_video is None else str(current_video),
        "current_stage": stage,
        "elapsed_time_sec": elapsed_sec,
    }
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
