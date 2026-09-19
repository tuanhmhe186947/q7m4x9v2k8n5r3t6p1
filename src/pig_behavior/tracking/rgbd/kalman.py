# ruff: noqa
"""Constant-velocity Kalman Filter for BEV (Bird's-Eye-View) tracking.

# ruff: noqa

State vector ``[x, y, vx, vy]`` in world/BEV coordinates (metres).
Measurement vector ``[x, y]``.

This is a minimal NumPy-only implementation so the project does not
need an external Kalman library dependency.  ``scipy`` is *not* required
for the filter itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig


@dataclass
class BEVKalmanFilter:
    """Four-state constant-velocity Kalman Filter operating in BEV metres.

    Attributes
    ----------
    x : NDArray[np.float64]
        State mean ``[x, y, vx, vy]``, shape ``(4,)``.
    P : NDArray[np.float64]
        State covariance ``(4, 4)``.
    F : NDArray[np.float64]
        State transition matrix ``(4, 4)``.
    H : NDArray[np.float64]
        Measurement matrix ``(2, 4)``.
    Q : NDArray[np.float64]
        Process noise ``(4, 4)``.
    R : NDArray[np.float64]
        Measurement noise ``(2, 2)``.
    """

    x: NDArray[np.float64]
    P: NDArray[np.float64]
    F: NDArray[np.float64]
    H: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]


def create_bev_kalman(
    initial_xy: NDArray[np.float64],
    cfg: RGBDTrackingConfig,
) -> BEVKalmanFilter:
    """Create a new BEV Kalman Filter initialised at ``initial_xy``.

    Parameters
    ----------
    initial_xy:
        ``(2,)`` initial position in BEV metres.
    cfg:
        Configuration for process/measurement noise tuning.
    """
    assert initial_xy.shape == (2,), f"Expected shape (2,) for initial_xy, got {initial_xy.shape}"
    dt = 1.0  # normalised frame step

    # State transition: constant velocity
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Measurement matrix: observe position only
    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    # Process noise — continuous white-noise acceleration model
    q = cfg.kf_process_std ** 2
    Q = q * np.array(
        [
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ],
        dtype=np.float64,
    )

    # Measurement noise
    r = cfg.kf_measurement_std ** 2
    R = r * np.eye(2, dtype=np.float64)

    # Initial state
    x = np.array(
        [initial_xy[0], initial_xy[1], 0.0, 0.0],
        dtype=np.float64,
    )

    # Initial covariance — large uncertainty on velocity
    P = np.diag([r, r, 1.0, 1.0]).astype(np.float64)

    return BEVKalmanFilter(x=x, P=P, F=F, H=H, Q=Q, R=R)


def predict_bev(kf: BEVKalmanFilter) -> NDArray[np.float64]:
    """Predict the next state.  Returns the predicted BEV position ``(2,)``."""
    assert kf.x.shape == (4,), f"Expected Kalman state vector shape (4,), got {kf.x.shape}"
    assert kf.F.shape == (4, 4), f"Expected transition matrix shape (4, 4), got {kf.F.shape}"
    assert kf.P.shape == (4, 4), f"Expected covariance matrix shape (4, 4), got {kf.P.shape}"
    assert kf.Q.shape == (4, 4), f"Expected process noise shape (4, 4), got {kf.Q.shape}"

    kf.x = kf.F @ kf.x
    kf.P = kf.F @ kf.P @ kf.F.T + kf.Q
    return kf.x[:2].copy()


def update_bev(kf: BEVKalmanFilter, measurement_xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Correct the state with a BEV measurement.  Returns updated position ``(2,)``."""
    z = np.asarray(measurement_xy, dtype=np.float64)
    assert z.shape == (2,), f"Expected shape (2,) for measurement_xy, got {z.shape}"
    assert kf.x.shape == (4,), f"Expected Kalman state vector shape (4,), got {kf.x.shape}"
    assert kf.H.shape == (2, 4), f"Expected measurement matrix shape (2, 4), got {kf.H.shape}"
    assert kf.P.shape == (4, 4), f"Expected covariance matrix shape (4, 4), got {kf.P.shape}"
    assert kf.R.shape == (2, 2), f"Expected measurement noise shape (2, 2), got {kf.R.shape}"

    y = z - kf.H @ kf.x  # innovation
    S = kf.H @ kf.P @ kf.H.T + kf.R  # innovation covariance
    K = kf.P @ kf.H.T @ np.linalg.inv(S)  # Kalman gain
    kf.x = kf.x + K @ y
    I = np.eye(4, dtype=np.float64)
    kf.P = (I - K @ kf.H) @ kf.P
    return kf.x[:2].copy()


def bev_position(kf: BEVKalmanFilter) -> NDArray[np.float64]:
    """Return the current BEV position estimate ``(2,)``."""
    assert kf.x.shape == (4,), f"Expected Kalman state vector shape (4,), got {kf.x.shape}"
    return kf.x[:2].copy()


def bev_velocity(kf: BEVKalmanFilter) -> NDArray[np.float64]:
    """Return the current BEV velocity estimate ``(2,)``."""
    assert kf.x.shape == (4,), f"Expected Kalman state vector shape (4,), got {kf.x.shape}"
    return kf.x[2:4].copy()


__all__ = [
    "BEVKalmanFilter",
    "bev_position",
    "bev_velocity",
    "create_bev_kalman",
    "predict_bev",
    "update_bev",
]
