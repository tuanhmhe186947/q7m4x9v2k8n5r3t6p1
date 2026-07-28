"""Independent elementary acceleration calculator for scientific tests.

This module intentionally imports no production feature, schema, pair-validity,
aggregation, exporter, or model code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def calculate_acceleration_reference(
    *,
    timestamps: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    temporal_units: Sequence[str] | None = None,
    actors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate velocity, direction, and acceleration from definitions."""

    row_count = len(timestamps)
    if len(x) != row_count or len(y) != row_count:
        raise ValueError("timestamps and positions must have equal lengths")
    units = list(temporal_units or ["unit-a"] * row_count)
    actor_keys = list(actors or ["actor-a"] * row_count)
    if len(units) != row_count or len(actor_keys) != row_count:
        raise ValueError("authority vectors must match timestamps")

    vx = np.full(row_count, np.nan, dtype="float64")
    vy = np.full(row_count, np.nan, dtype="float64")
    speed = np.full(row_count, np.nan, dtype="float64")
    velocity_time = np.full(row_count, np.nan, dtype="float64")
    velocity_valid = np.zeros(row_count, dtype=bool)
    direction = np.full(row_count, np.nan, dtype="float64")
    direction_valid = np.zeros(row_count, dtype=bool)

    for index in range(1, row_count):
        delta_t = float(timestamps[index]) - float(timestamps[index - 1])
        same_authority = (
            units[index] == units[index - 1]
            and actor_keys[index] == actor_keys[index - 1]
        )
        if not same_authority or not math.isfinite(delta_t) or delta_t <= 0:
            continue
        vx[index] = (float(x[index]) - float(x[index - 1])) / delta_t
        vy[index] = (float(y[index]) - float(y[index - 1])) / delta_t
        speed[index] = math.hypot(vx[index], vy[index])
        velocity_time[index] = (
            float(timestamps[index]) + float(timestamps[index - 1])
        ) / 2.0
        velocity_valid[index] = True
        if speed[index] > 0:
            direction[index] = math.atan2(vy[index], vx[index])
            direction_valid[index] = True

    acceleration_delta_t = np.full(row_count, np.nan, dtype="float64")
    tangential = np.full(row_count, np.nan, dtype="float64")
    ax = np.full(row_count, np.nan, dtype="float64")
    ay = np.full(row_count, np.nan, dtype="float64")
    vector_magnitude = np.full(row_count, np.nan, dtype="float64")
    acceleration_valid = np.zeros(row_count, dtype=bool)
    direction_change = np.full(row_count, np.nan, dtype="float64")

    for index in range(1, row_count):
        same_authority = (
            units[index] == units[index - 1]
            and actor_keys[index] == actor_keys[index - 1]
        )
        if velocity_valid[index] and velocity_valid[index - 1] and same_authority:
            delta_t = velocity_time[index] - velocity_time[index - 1]
            if math.isfinite(delta_t) and delta_t > 0:
                acceleration_delta_t[index] = delta_t
                tangential[index] = (
                    speed[index] - speed[index - 1]
                ) / delta_t
                ax[index] = (vx[index] - vx[index - 1]) / delta_t
                ay[index] = (vy[index] - vy[index - 1]) / delta_t
                vector_magnitude[index] = math.hypot(ax[index], ay[index])
                acceleration_valid[index] = True
        if direction_valid[index] and direction_valid[index - 1] and same_authority:
            raw = direction[index] - direction[index - 1]
            direction_change[index] = (raw + math.pi) % (2 * math.pi) - math.pi

    return pd.DataFrame(
        {
            "vx_n_per_second": vx,
            "vy_n_per_second": vy,
            "speed_n_per_second": speed,
            "velocity_midpoint_time_sec": velocity_time,
            "velocity_valid": velocity_valid,
            "direction_rad": direction,
            "direction_valid": direction_valid,
            "direction_change_rad": direction_change,
            "acceleration_delta_t_sec": acceleration_delta_t,
            "tangential_acceleration_n_per_second2": tangential,
            "ax_n_per_second2": ax,
            "ay_n_per_second2": ay,
            "acceleration_vector_magnitude_n_per_second2": vector_magnitude,
            "acceleration_valid": acceleration_valid,
        }
    )
