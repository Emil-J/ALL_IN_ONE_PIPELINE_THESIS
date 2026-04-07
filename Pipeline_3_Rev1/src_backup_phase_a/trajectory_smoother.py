"""
Module 12 — Trajectory Smoother.

Post-processing utilities: Kalman smoothing, moving average,
outlier detection, and gap filling.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ─── 12.1  smooth_trajectory ─────────────────────────────────

def smooth_trajectory(positions: List[Tuple[float, float, float, float]],
                      method: str = "kalman"
                      ) -> List[Tuple[float, float, float, float]]:
    """
    Smooth a trajectory of (lat, lon, heading, timestamp) tuples.

    Methods:
        'kalman'         — simple 1-D Kalman smoother per coordinate
        'moving_average' — sliding window average
        'spline'         — cubic spline interpolation
    """
    if len(positions) < 3:
        return list(positions)

    lats = np.array([p[0] for p in positions])
    lons = np.array([p[1] for p in positions])
    hdgs = np.array([p[2] for p in positions])
    ts   = np.array([p[3] for p in positions])

    if method == "kalman":
        lats_s = _kalman_smooth_1d(lats)
        lons_s = _kalman_smooth_1d(lons)
        hdgs_s = _circular_smooth(hdgs, window=5)
    elif method == "moving_average":
        lats_s = _moving_average(lats, window=5)
        lons_s = _moving_average(lons, window=5)
        hdgs_s = _circular_smooth(hdgs, window=5)
    elif method == "spline":
        from scipy.interpolate import CubicSpline
        cs_lat = CubicSpline(ts, lats)
        cs_lon = CubicSpline(ts, lons)
        lats_s = cs_lat(ts)
        lons_s = cs_lon(ts)
        hdgs_s = _circular_smooth(hdgs, window=5)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return [(float(la), float(lo), float(h), float(t))
            for la, lo, h, t in zip(lats_s, lons_s, hdgs_s, ts)]


# ─── 12.2  detect_outliers ───────────────────────────────────

def detect_outliers(positions: List[Tuple[float, float, float, float]],
                    threshold_meters: float = 100.0) -> List[int]:
    """Return indices of frames whose jump from the previous frame exceeds *threshold_meters*."""
    outliers = []
    for i in range(1, len(positions)):
        d = _haversine(positions[i - 1][0], positions[i - 1][1],
                       positions[i][0], positions[i][1])
        if d > threshold_meters:
            outliers.append(i)
    return outliers


# ─── 12.3  fill_gaps ─────────────────────────────────────────

def fill_gaps(positions: List[Tuple[float, float, float, float]],
              expected_dt: float = 0.46
              ) -> List[Tuple[float, float, float, float]]:
    """
    Linearly interpolate missing frames (gaps > 1.5× expected_dt).
    """
    if len(positions) < 2:
        return list(positions)

    filled = [positions[0]]
    for i in range(1, len(positions)):
        prev = positions[i - 1]
        curr = positions[i]
        gap = curr[3] - prev[3]
        if gap > 1.5 * expected_dt:
            n_insert = max(int(round(gap / expected_dt)) - 1, 1)
            for j in range(1, n_insert + 1):
                frac = j / (n_insert + 1)
                lat = prev[0] + frac * (curr[0] - prev[0])
                lon = prev[1] + frac * (curr[1] - prev[1])
                hdg = prev[2] + frac * _angular_diff_smooth(prev[2], curr[2])
                t = prev[3] + frac * gap
                filled.append((lat, lon, hdg % 360, t))
        filled.append(curr)
    return filled


# ─── Internal helpers ─────────────────────────────────────────

def _moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def _kalman_smooth_1d(arr: np.ndarray) -> np.ndarray:
    """Simple forward-backward Kalman smoother for a 1-D signal."""
    n = len(arr)
    # Forward pass
    x_fwd = np.zeros(n)
    p_fwd = np.zeros(n)
    x_fwd[0] = arr[0]
    p_fwd[0] = 1.0
    q = 1e-5   # process noise
    r = 1e-4   # measurement noise
    for i in range(1, n):
        x_pred = x_fwd[i - 1]
        p_pred = p_fwd[i - 1] + q
        k = p_pred / (p_pred + r)
        x_fwd[i] = x_pred + k * (arr[i] - x_pred)
        p_fwd[i] = (1 - k) * p_pred

    # Backward pass
    x_bwd = np.zeros(n)
    p_bwd = np.zeros(n)
    x_bwd[-1] = arr[-1]
    p_bwd[-1] = 1.0
    for i in range(n - 2, -1, -1):
        x_pred = x_bwd[i + 1]
        p_pred = p_bwd[i + 1] + q
        k = p_pred / (p_pred + r)
        x_bwd[i] = x_pred + k * (arr[i] - x_pred)
        p_bwd[i] = (1 - k) * p_pred

    # Combine
    smoothed = np.zeros(n)
    for i in range(n):
        w_fwd = 1.0 / max(p_fwd[i], 1e-30)
        w_bwd = 1.0 / max(p_bwd[i], 1e-30)
        smoothed[i] = (w_fwd * x_fwd[i] + w_bwd * x_bwd[i]) / (w_fwd + w_bwd)
    return smoothed


def _circular_smooth(headings_deg: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth circular heading data via vector averaging."""
    n = len(headings_deg)
    result = np.zeros(n)
    rad = np.deg2rad(headings_deg)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        s = np.mean(np.sin(rad[lo:hi]))
        c = np.mean(np.cos(rad[lo:hi]))
        result[i] = np.degrees(np.arctan2(s, c)) % 360
    return result


def _angular_diff_smooth(a: float, b: float) -> float:
    d = (b - a) % 360
    return d if d <= 180 else d - 360
