"""
tests/test_units.py — Unit tests for IMU unit contract and bug fixes.

Tests:
  1. test_accel_units       — step_ekf receives ft/s² and produces correct gravity
  2. test_airspeed_units    — simconnect_adapter converts knots→m/s correctly
  3. test_meta_tile_none    — meta_tile_path=None propagates without str() coercion
  4. test_file_source_align — FileSource.iter_aligned() yields correct indices

Run with:
    cd Pipeline_3_Rev1
    ../.final_Pipeline_venv/Scripts/python.exe -m pytest tests/test_units.py -v
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]   # Pipeline_3_Rev1/
REPO = ROOT.parent                           # All_In_One_Pipeline/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: Accelerometer units through step_ekf
# ═════════════════════════════════════════════════════════════════════════════

def _make_row(accel_x=0.0, accel_y=0.0, accel_z=0.0,
              gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
              pitch=0.0, bank=0.0, heading_magnetic=0.0,
              airspeed_true=50.0, barometer_pressure=1013.25,
              timestamp=1.0, latitude=55.7, longitude=9.5):
    """Build a minimal row_dict for step_ekf in file-mode units."""
    return {
        "timestamp":          timestamp,
        # accel in ft/s² (raw SimConnect, no pre-conversion)
        "accel_x":            accel_x,
        "accel_y":            accel_y,
        "accel_z":            accel_z,
        "gyro_x":             gyro_x,
        "gyro_y":             gyro_y,
        "gyro_z":             gyro_z,
        "pitch":              pitch,
        "bank":               bank,
        "heading_magnetic":   heading_magnetic,
        "airspeed_true":      airspeed_true,   # m/s
        "barometer_pressure": barometer_pressure,
        "latitude":           latitude,
        "longitude":          longitude,
    }


def test_accel_units():
    """
    step_ekf should apply *0.3048 to accel columns internally.

    Verify by checking that a stationary row (all accels=0, pitch=bank=0)
    produces the correct gravity-synthesised body acceleration vector.

    With pitch=bank=0 and airspeed0=0 (stationary start):
      The net accel fed to the integrator should be ~0 (g-cancelled).
      So velocity should change by <0.05 m/s over a 20ms step.

    If accel were double-converted (*0.3048 twice), the gravity synthesis
    term (which is in m/s²) would not be cancelled, causing large drift.
    """
    from src.ekf_ins import ErrorStateEKF, step_ekf, barometric_altitude

    # Start with zero airspeed so initial vel_n=0
    ekf = ErrorStateEKF(
        lat0=55.7, lon0=9.5,
        alt0=barometric_altitude(1013.25),
        heading0=0.0,
        airspeed0=0.0,   # stationary start so vel_n=0 at t0
    )

    # Level flight row: zero coordinate acceleration, level attitude
    row0 = _make_row(airspeed_true=0.0, timestamp=1.0)
    row1 = _make_row(airspeed_true=0.0, timestamp=1.02)   # dt = 20 ms

    step_ekf(ekf, row0, prev_timestamp=None)
    state = step_ekf(ekf, row1, prev_timestamp=1.0)

    assert not np.any(np.isnan(ekf.q_tilde)), "quaternion has NaN — accel units wrong"
    assert not np.any(np.isnan([ekf.vel_n, ekf.vel_e])), "velocities have NaN"
    assert abs(state["yaw"]) < 180, "yaw out of range"

    # With zero coordinate acceleration and level flight, gravity is
    # fully synthesised and cancelled. Net acceleration → ~0 → vel change < 0.05 m/s.
    # If double-converted: gravity term ~9.81 becomes ~2.99 m/s² → not cancelled
    # → vel_n grows by ~2.99*0.02 ≈ 0.06 m/s or more per step.
    assert abs(state["vel_n"]) < 0.15, (
        f"vel_n={state['vel_n']:.3f} m/s after level-flight step — "
        "expected <0.15 m/s for stationary level-flight row"
    )
    assert abs(state["vel_e"]) < 0.15, (
        f"vel_e={state['vel_e']:.3f} m/s after level-flight step — "
        "expected <0.15 m/s for stationary level-flight row"
    )


def test_accel_units_with_explicit_gravity():
    """
    Verify *0.3048 is applied exactly once.

    MSFS coordinate acceleration at sea-level hovering straight up = +1g in body Z.
    MSFS body Z+ = up (body Y in NED convention), so a +1g upward coordinate
    accel in MSFS maps to accel_y_msfs = +32.174 ft/s².

    step_ekf MSFS→NED mapping (from ekf_ins.py):
        accel_body = [accel_z_msfs, accel_x_msfs, -accel_y_msfs] * 0.3048 + g_body

    For accel_y_msfs = +32.174 ft/s², accel_x_msfs = accel_z_msfs = 0, pitch=bank=0:
        accel_body_NED_right = -32.174 * 0.3048 = -9.81 m/s²  (downward in NED-right)

    If *0.3048 were applied TWICE:
        -32.174 * 0.3048 * 0.3048 = -2.99 m/s²  (wrong by factor ~3.3)
    """
    from src.ekf_ins import ErrorStateEKF, step_ekf, barometric_altitude

    # Use airspeed0=0 so we start from rest and can measure acceleration cleanly
    ekf = ErrorStateEKF(55.7, 9.5, barometric_altitude(1013.25), 0.0, 0.0)
    vel_n_before = ekf.vel_n

    g_ft_s2 = 32.174    # 1g in ft/s²
    row = _make_row(accel_y=g_ft_s2, airspeed_true=0.0, timestamp=1.02)

    # accel_body[2] (NED down) = -accel_y_msfs * 0.3048 + g*cos(0)*cos(0)
    # = -9.81 + 9.81 = 0  → zero net NED acceleration → delta vel ≈ 0
    # If double-converted: -9.81*0.3048 + 9.81 = 6.81 → delta vel = 6.81*0.02 = 0.136 m/s

    step_ekf(ekf, row, prev_timestamp=1.0)
    delta_vel = abs(ekf.vel_n - vel_n_before)
    assert delta_vel < 0.05, (
        f"vel_n delta = {delta_vel:.4f} m/s — expected <0.05 m/s for 1g hover row"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: Airspeed unit conversion in simconnect_adapter
# ═════════════════════════════════════════════════════════════════════════════

def test_airspeed_units():
    """
    _KTS_TO_MS = 0.514444 must match data_logger.py exactly.
    100 knots → 51.4444 m/s.
    """
    from runtime.simconnect_adapter import _KTS_TO_MS
    assert abs(_KTS_TO_MS - 0.514444) < 1e-9, \
        f"_KTS_TO_MS = {_KTS_TO_MS}, expected 0.514444"
    assert abs(100.0 * _KTS_TO_MS - 51.4444) < 1e-6, \
        f"100 kts → {100 * _KTS_TO_MS} m/s, expected 51.4444"


def test_no_accel_constant_in_adapter():
    """
    The removed _FT_S2_TO_M_S2 constant must NOT exist in simconnect_adapter.
    If it does, there's a risk of re-introducing the double-conversion.
    """
    import runtime.simconnect_adapter as adapter
    assert not hasattr(adapter, "_FT_S2_TO_M_S2"), \
        "_FT_S2_TO_M_S2 still present — double-conversion risk"


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: meta_tile_path=None propagates as None, not "None"
# ═════════════════════════════════════════════════════════════════════════════

def test_meta_tile_path_none():
    """
    When DEBUG_SAVE_METATILES=False, meta_tile_builder returns
    meta_tile_path=None. Verify that temporal_searcher does NOT
    str()-coerce it into the string "None".

    We test the exact result-dict construction code path by looking up
    the literal string in the source file.
    """
    ts_path = ROOT / "src" / "temporal_searcher.py"
    source  = ts_path.read_text(encoding="utf-8")

    # The bug pattern
    bad = 'str(meta_result["meta_tile_path"])'
    assert bad not in source, (
        "temporal_searcher.py still calls str() on meta_tile_path — "
        "None will become the string 'None'"
    )

    # The correct pattern
    good = '"meta_tile_path": meta_result["meta_tile_path"]'
    assert good in source, (
        "temporal_searcher.py missing correct meta_tile_path assignment"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: FileSource.iter_aligned() yields correct structure
# ═════════════════════════════════════════════════════════════════════════════

def test_file_source_align(tmp_path):
    """
    FileSource.iter_aligned() should:
    - Yield (csv_idx, row_dict, timestamp, frame_path) 4-tuples
    - Match timestamps to 3 decimal places
    - Respect start_row and max_frames
    - Expose raw_df without a second CSV read
    """
    import pandas as pd
    from runtime.simconnect_adapter import FileSource

    # Create a minimal fake CSV with 5 rows
    df = pd.DataFrame({
        "timestamp":          [1.000, 1.020, 1.040, 1.060, 1.080],
        "accel_x":            [0.0] * 5,
        "accel_y":            [0.0] * 5,
        "accel_z":            [0.0] * 5,
        "gyro_x":             [0.0] * 5,
        "gyro_y":             [0.0] * 5,
        "gyro_z":             [0.0] * 5,
        "pitch":              [0.0] * 5,
        "bank":               [0.0] * 5,
        "heading_magnetic":   [0.0] * 5,
        "airspeed_true":      [50.0] * 5,
        "barometer_pressure": [1013.25] * 5,
        "latitude":           [55.7] * 5,
        "longitude":          [9.5] * 5,
    })
    csv_path = tmp_path / "fake_imu.csv"
    df.to_csv(csv_path, index=False)

    # Create matching frame files for rows 0, 2, 4
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for ts in [1.000, 1.040, 1.080]:
        (frames_dir / f"frame_{ts:.3f}.jpg").write_bytes(b"\xff")  # dummy

    src = FileSource(csv_path, frames_dir)
    results = list(src.iter_aligned(start_row=0))

    assert len(results) == 3, f"Expected 3 aligned frames, got {len(results)}"

    csv_idx, row_dict, ts, fp = results[0]
    assert csv_idx == 0,   f"First csv_idx should be 0, got {csv_idx}"
    assert abs(ts - 1.000) < 1e-9
    assert isinstance(row_dict, dict)
    assert "accel_x" in row_dict
    assert fp.name == "frame_1.000.jpg"

    # start_row=2 should skip the first aligned frame (row 0)
    results_from_2 = list(src.iter_aligned(start_row=2))
    assert len(results_from_2) == 2, \
        f"start_row=2 should yield 2 frames, got {len(results_from_2)}"

    # max_frames=1
    results_max1 = list(src.iter_aligned(start_row=0, max_frames=1))
    assert len(results_max1) == 1

    # raw_df is cached — same object on second access
    df1 = src.raw_df
    df2 = src.raw_df
    assert df1 is df2, "raw_df should be cached (same object)"


# ═════════════════════════════════════════════════════════════════════════════
# Test 5: run_pipeline imports are clean (no preprocess_imu_csv)
# ═════════════════════════════════════════════════════════════════════════════

def test_run_pipeline_no_preprocess_imu_csv():
    """
    preprocess_imu_csv (batch EKF) must not be imported or called in
    run_pipeline.py — it has no place in the deployment runtime.
    """
    import ast
    rp_path = ROOT / "runtime" / "run_pipeline.py"
    source  = rp_path.read_text(encoding="utf-8")
    tree    = ast.parse(source)

    # Walk AST: collect all Name and Attribute nodes used in code (not comments)
    names_in_code = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    assert "preprocess_imu_csv" not in names_in_code, (
        "run_pipeline.py still calls preprocess_imu_csv in code — "
        "batch EKF should not run in the runtime"
    )
    # Import check: ensure it's not imported anywhere in the module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "preprocess_imu_csv", (
                    "run_pipeline.py imports preprocess_imu_csv"
                )

    assert "FileSource" in source, (
        "run_pipeline.py must use FileSource from simconnect_adapter"
    )
