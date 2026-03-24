"""Unit tests for Module 12 — Trajectory Smoother."""

import pytest
import numpy as np
from src.trajectory_smoother import (
    smooth_trajectory,
    detect_outliers,
    fill_gaps,
)


def _make_positions(n=20, noise_std=0.00005):
    """Generate a simple linear trajectory with noise."""
    rng = np.random.default_rng(42)
    return [
        (55.7 + 0.0001 * i + rng.normal(0, noise_std),
         9.5 + 0.0001 * i + rng.normal(0, noise_std),
         180.0,
         i * 0.46)
        for i in range(n)
    ]


def test_smooth_kalman_preserves_length():
    pos = _make_positions()
    smoothed = smooth_trajectory(pos, method="kalman")
    assert len(smoothed) == len(pos)


def test_smooth_moving_average_preserves_length():
    pos = _make_positions()
    smoothed = smooth_trajectory(pos, method="moving_average")
    assert len(smoothed) == len(pos)


def test_smooth_short_input():
    pos = _make_positions(2)
    smoothed = smooth_trajectory(pos, method="kalman")
    assert len(smoothed) == 2


def test_detect_outliers_finds_jump():
    pos = _make_positions(10, noise_std=0)
    # Insert a massive jump at index 5
    lat, lon, h, t = pos[5]
    pos[5] = (lat + 0.1, lon + 0.1, h, t)  # ~10 km jump
    outliers = detect_outliers(pos, threshold_meters=100)
    assert 5 in outliers


def test_detect_outliers_clean():
    pos = _make_positions(10, noise_std=0)
    outliers = detect_outliers(pos, threshold_meters=100)
    assert len(outliers) == 0


def test_fill_gaps_inserts_frames():
    pos = [
        (55.7, 9.5, 180, 0.0),
        (55.7001, 9.5001, 180, 0.46),
        # Large gap
        (55.7005, 9.5005, 180, 2.30),
    ]
    filled = fill_gaps(pos, expected_dt=0.46)
    assert len(filled) > len(pos)


def test_fill_gaps_no_gaps():
    pos = _make_positions(5, noise_std=0)
    filled = fill_gaps(pos, expected_dt=0.46)
    assert len(filled) == len(pos)
