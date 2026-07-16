"""Prediction-invariant runtime telemetry helpers for tracking."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from typing import Any

from pig_behavior.tracking.constants import (
    TRACKING_ASSOCIATION_PHASES,
    TRACKING_FLOAT_TELEMETRY_KEYS,
    TRACKING_INTEGER_TELEMETRY_KEYS,
    TRACKING_TEXT_TELEMETRY_KEYS,
    TRACKING_TIMING_STAGES,
)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def record_timing_sample(
    runtime: Any,
    stage: str,
    elapsed_seconds: float,
) -> None:
    """Record one non-negative host wall-time sample."""
    if stage not in TRACKING_TIMING_STAGES:
        raise ValueError(f"Unknown tracking timing stage: {stage}")
    samples = runtime.timing_samples_seconds.setdefault(stage, [])
    samples.append(max(0.0, float(elapsed_seconds)))


def record_association_phase(runtime: Any | None, phase_name: str) -> None:
    """Count one association phase without changing its decisions."""
    if runtime is None or phase_name not in TRACKING_ASSOCIATION_PHASES:
        return
    key = f"association_phase_{phase_name}_calls"
    runtime.telemetry[key] = int(runtime.telemetry.get(key, 0)) + 1


def record_association_event(runtime: Any | None, event_name: object) -> None:
    """Aggregate association outcomes independently from debug logging."""
    if runtime is None:
        return
    name = str(event_name or "")
    if name == "assignment_accept":
        key = "association_assignments_accepted"
    elif name.startswith("assignment_reject"):
        key = "association_assignments_rejected"
    elif name.startswith("assignment_prefer"):
        key = "association_assignments_preferred"
    elif "hold" in name or name == "assignment_area_freeze":
        key = "association_assignments_held"
    else:
        return
    runtime.telemetry[key] = int(runtime.telemetry.get(key, 0)) + 1


def resolve_output_timing_contract(cfg: Any) -> tuple[str, int]:
    """Return the truthful output-causality contract and fixed delay."""
    if cfg.mode == "realtime" and cfg.realtime_motion_pair_stabilizer:
        return "post_video_global_graph", -1
    if cfg.enable_offline_smoothing:
        return "post_video_offline", -1
    return "causal_framewise", 0


def peak_process_rss_bytes() -> int:
    """Return peak resident memory for the current process when supported."""
    if os.name == "nt":
        return _windows_peak_process_rss_bytes()
    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return 0
    if sys.platform == "darwin":
        return peak_rss
    return peak_rss * 1024


def _windows_peak_process_rss_bytes() -> int:
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        ok = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if ok else 0
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def summarize_tracking_telemetry(source: Any) -> dict[str, int | float | str]:
    """Build the stable counter, timing, delay, and memory schema."""
    raw: Mapping[str, object] = getattr(source, "telemetry", {})
    summary: dict[str, int | float | str] = {
        key: int(raw.get(key, 0)) for key in TRACKING_INTEGER_TELEMETRY_KEYS
    }
    summary.update(
        {key: float(raw.get(key, 0.0)) for key in TRACKING_FLOAT_TELEMETRY_KEYS}
    )
    summary.update(
        {
            key: value if isinstance(value, str) else ""
            for key in TRACKING_TEXT_TELEMETRY_KEYS
            for value in (raw.get(key, ""),)
        }
    )

    timing_samples: Mapping[str, list[float]] = getattr(
        source,
        "timing_samples_seconds",
        {},
    )
    for stage in TRACKING_TIMING_STAGES:
        values = list(timing_samples.get(stage, []))
        total_seconds = sum(values)
        summary[f"{stage}_time_ms_total"] = total_seconds * 1000.0
        summary[f"{stage}_time_ms_mean"] = (
            total_seconds * 1000.0 / len(values) if values else 0.0
        )
        summary[f"{stage}_time_ms_p50"] = _percentile(values, 0.50) * 1000.0
        summary[f"{stage}_time_ms_p95"] = _percentile(values, 0.95) * 1000.0

    frames = int(summary["frames_processed"])
    frame_seconds = float(summary["frame_time_ms_total"]) / 1000.0
    postprocess_seconds = float(summary["postprocess_time_ms_total"]) / 1000.0
    summary["tracking_loop_effective_fps"] = (
        frames / frame_seconds if frames and frame_seconds > 0.0 else 0.0
    )
    timed_seconds = frame_seconds + postprocess_seconds
    summary["effective_fps"] = (
        frames / timed_seconds if frames and timed_seconds > 0.0 else 0.0
    )

    delay_frames = int(summary["declared_delay_frames"])
    source_fps = float(summary["source_fps"])
    summary["declared_delay_ms"] = (
        delay_frames * 1000.0 / source_fps
        if delay_frames >= 0 and source_fps > 0.0
        else -1.0
    )
    return summary


__all__ = [
    "peak_process_rss_bytes",
    "record_association_event",
    "record_association_phase",
    "record_timing_sample",
    "resolve_output_timing_contract",
    "summarize_tracking_telemetry",
]
