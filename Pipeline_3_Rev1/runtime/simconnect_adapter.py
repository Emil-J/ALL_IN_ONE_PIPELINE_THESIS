"""
runtime/simconnect_adapter.py — Data source abstraction for run_pipeline.py.

Two classes:
  FileSource          — reads pre-recorded IMU CSV + frame directory
  SimConnectLiveSource — background-thread SimConnect polling at ~50Hz,
                         non-blocking get_latest_row() / get_latest_frame()

The visual pipeline loop (~1s/frame) calls get_latest_row() / get_latest_frame()
without blocking the background IMU acquisition thread. This keeps the EKF
running at full rate regardless of how long feature matching takes.

=== UNIT CONTRACT ===
Both sources produce row dicts matching the units step_ekf() expects
(same as MSFS2020_IMU_Pipeline/data_logger.py CSV format):

  accel_x/y/z        — ft/s²  (raw; step_ekf applies *0.3048 internally)
  gyro_x/y/z         — rad/s
  heading_magnetic   — radians  (SimConnect _DEGREES_ vars are actually rad)
  heading            — radians  (true heading, PLANE_HEADING_DEGREES_TRUE)
  pitch, bank        — radians
  airspeed_true      — m/s     (converted here: kts * 0.514444)
  airspeed_indicated — m/s     (converted here: kts * 0.514444)
  ground_velocity    — m/s     (converted here: kts * 0.514444)
  vertical_speed     — m/s     (converted here: ft/min * 0.00508)
  barometer_pressure — mbar    (BAROMETER_PRESSURE = ambient pressure,
                                 NOT KOHLSMAN_SETTING_MB which is the QNH dial)
  latitude/longitude — degrees (GPS ground truth from SimConnect)
  altitude           — feet    (GPS altitude, PLANE_ALTITUDE)
  pressure_altitude  — metres  (PRESSURE_ALTITUDE; Python SimConnect returns SI, NOT feet)
  magnetic_compass   — degrees (MAGNETIC_COMPASS)
  timestamp          — seconds

DO NOT convert accel ft/s²→m/s² in this file. step_ekf() does it.
"""

import copy
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

# knots → m/s  (matches data_logger.py: airspeed_true = kts * 0.514444)
_KTS_TO_MS = 0.514444


# ═════════════════════════════════════════════════════════════════════════════
# FileSource
# ═════════════════════════════════════════════════════════════════════════════

class FileSource:
    """
    Iterates over aligned (csv_row, timestamp, frame_path) tuples from a
    pre-recorded flight log.

    Usage:
        src = FileSource(imu_csv_path, frames_dir)
        for csv_idx, row_dict, timestamp, frame_path in src.iter_aligned():
            ...
    """

    def __init__(self, imu_csv_path: Path, frames_dir: Path):
        self._imu_csv    = Path(imu_csv_path)
        self._frames_dir = Path(frames_dir)
        self._raw_df     = None

    @property
    def raw_df(self) -> pd.DataFrame:
        if self._raw_df is None:
            self._raw_df = pd.read_csv(self._imu_csv)
        return self._raw_df

    def iter_aligned(
        self,
        start_row: int = 0,
        max_frames: Optional[int] = None,
    ) -> Iterator[Tuple[int, dict, float, Path]]:
        """
        Yields (csv_idx, row_dict, timestamp, frame_path) for every IMU row
        that has a matching frame file (rounded to 3 decimal places).
        """
        df = self.raw_df
        frame_files = sorted(self._frames_dir.glob("frame_*.jpg"))
        frame_map: dict = {}
        for fp in frame_files:
            ts_str = fp.stem.replace("frame_", "")
            try:
                frame_map[round(float(ts_str), 3)] = fp
            except ValueError:
                pass

        count = 0
        for idx in range(start_row, len(df)):
            row = df.iloc[idx]
            ts_rounded = round(row["timestamp"], 3)
            if ts_rounded in frame_map:
                if max_frames is not None and count >= max_frames:
                    break
                yield (idx, row.to_dict(), row["timestamp"],
                       frame_map[ts_rounded])
                count += 1


# ═════════════════════════════════════════════════════════════════════════════
# SimConnectLiveSource
# ═════════════════════════════════════════════════════════════════════════════

class SimConnectLiveSource:
    """
    Background-thread SimConnect poller.

    The daemon thread polls SimConnect at ~50Hz and captures a screen frame
    every CAPTURE_EVERY_N iterations (~5fps). Results are stored under a
    threading.Lock so the visual pipeline loop can call:
        row  = source.get_latest_row()    # non-blocking, returns dict or None
        img, fid = source.get_latest_frame()  # non-blocking, returns (array|None, id)

    SimConnect and mss are only imported inside connect() so that the file
    can be imported on machines without those packages installed.
    """

    POLL_HZ      = 50    # aspirational; actual rate is ~7 Hz (8 SimConnect calls × 18ms)
    CAPTURE_FPS  = 5     # target frame capture rate (time-based, not iteration-based)

    def __init__(self):
        self._lock                    = threading.Lock()
        self._latest_row              = None     # dict of IMU data
        self._latest_img              = None     # np.ndarray
        self._frame_id                = 0
        self._running                 = False
        self._thread: Optional[threading.Thread] = None
        self._last_capture_time       = 0.0    # perf_counter of last capture (rate-limiting)
        self._latest_frame_capture_ts = 0.0    # perf_counter when current frame was captured

        # SimConnect handles (set in connect())
        self._sm  = None
        self._aq  = None
        self._sct = None   # mss instance
        self._monitor = None

    def connect(self):
        """Import SimConnect, connect, detect MSFS window, start background thread."""
        try:
            from SimConnect import SimConnect, AircraftRequests
        except ImportError as e:
            raise ImportError(
                "SimConnect package not found. Install via: pip install SimConnect\n"
                f"Original error: {e}"
            ) from e
        try:
            import mss
        except ImportError as e:
            raise ImportError(
                "mss package not found. Install via: pip install mss\n"
                f"Original error: {e}"
            ) from e

        print("[SimConnectLiveSource] Connecting to MSFS 2020...")
        self._sm  = SimConnect()
        self._aq  = AircraftRequests(self._sm, _time=20)
        self._sct = mss.mss()
        self._monitor = self._detect_monitor()
        print(f"[SimConnectLiveSource] Connected. Capture region: {self._monitor}")

        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="simconnect-poller")
        self._thread.start()

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sm:
            try:
                self._sm.exit()
            except Exception:
                pass

    def get_latest_row(self) -> Optional[dict]:
        """Return a copy of the most recent IMU sample, or None."""
        with self._lock:
            return copy.copy(self._latest_row)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, float]:
        """Return (frame_array_copy_or_None, frame_id, capture_perf_counter_ts)."""
        with self._lock:
            img = self._latest_img
            fid = self._frame_id
            ts  = self._latest_frame_capture_ts
        return (img.copy() if img is not None else None), fid, ts

    # ── internal ──────────────────────────────────────────────────────────────

    def _detect_monitor(self) -> dict:
        """Try to find the MSFS window; fall back to primary monitor.

        Mirrors data_logger.py: tries FindWindow (exact title) first,
        then EnumWindows (partial match), then primary-monitor fallback.
        """
        try:
            import win32gui

            # 1. Exact title match (fastest, works for windowed mode)
            hwnd = win32gui.FindWindow(None, "Microsoft Flight Simulator")
            if hwnd and win32gui.IsWindowVisible(hwnd):
                _, _, w, h = win32gui.GetClientRect(hwnd)
                left, top  = win32gui.ClientToScreen(hwnd, (0, 0))
                return {"top": top, "left": left, "width": w, "height": h}

            # 2. Partial title match (handles version suffixes / exclusive fullscreen)
            handles = []
            def _cb(h, acc):
                if win32gui.IsWindowVisible(h):
                    if "flight simulator" in win32gui.GetWindowText(h).lower():
                        acc.append(h)
            win32gui.EnumWindows(_cb, handles)
            if handles:
                hwnd = handles[0]
                _, _, w, h = win32gui.GetClientRect(hwnd)
                left, top  = win32gui.ClientToScreen(hwnd, (0, 0))
                return {"top": top, "left": left, "width": w, "height": h}
        except Exception:
            pass
        # 3. Fallback — capture full primary screen
        return self._sct.monitors[1]

    def _poll_loop(self):
        aq               = self._aq
        interval         = 1.0 / self.POLL_HZ
        capture_interval = 1.0 / self.CAPTURE_FPS
        iter_n           = 0
        SLOW_N           = 5   # refresh slow vars every 5 iters

        # Pre-initialize with None so all keys are always present in row dicts
        # even before the first slow poll — mirrors data_logger.py slow_vars init.
        slow_cache = {
            "heading_magnetic":   None,
            "heading":            None,   # true heading, radians
            "airspeed_true":      None,   # m/s
            "airspeed_indicated": None,   # m/s
            "ground_velocity":    None,   # m/s
            "vertical_speed":     None,   # m/s
            "barometer_pressure": None,   # mbar (ambient pressure)
            "latitude":           None,   # degrees (GPS ground truth)
            "longitude":          None,   # degrees (GPS ground truth)
            "altitude":           None,   # feet
            "pressure_altitude":  None,   # feet
            "magnetic_compass":   None,   # degrees
        }

        while self._running:
            t_start = time.perf_counter()

            try:
                # ── Tier 1: critical IMU vars every iteration ─────────────
                accel_x = aq.get("ACCELERATION_BODY_X")
                accel_y = aq.get("ACCELERATION_BODY_Y")
                accel_z = aq.get("ACCELERATION_BODY_Z")
                gyro_x  = aq.get("ROTATION_VELOCITY_BODY_X")
                gyro_y  = aq.get("ROTATION_VELOCITY_BODY_Y")
                gyro_z  = aq.get("ROTATION_VELOCITY_BODY_Z")
                pitch   = aq.get("PLANE_PITCH_DEGREES")
                bank    = aq.get("PLANE_BANK_DEGREES")

                # ── Tier 2: slow vars every SLOW_N iterations ─────────────
                if iter_n % SLOW_N == 0:
                    slow_cache["heading_magnetic"] = aq.get(
                        "PLANE_HEADING_DEGREES_MAGNETIC")   # radians (_DEGREES_ = rad in SimConnect)
                    slow_cache["heading"] = aq.get(
                        "PLANE_HEADING_DEGREES_TRUE")        # radians (true heading)
                    _tas_kts = aq.get("AIRSPEED_TRUE")      # knots
                    slow_cache["airspeed_true"] = (
                        _tas_kts * _KTS_TO_MS if _tas_kts is not None else None)  # → m/s
                    _ias_kts = aq.get("AIRSPEED_INDICATED") # knots
                    slow_cache["airspeed_indicated"] = (
                        _ias_kts * _KTS_TO_MS if _ias_kts is not None else None)  # → m/s
                    _gs_kts  = aq.get("GROUND_VELOCITY")    # knots
                    slow_cache["ground_velocity"] = (
                        _gs_kts * _KTS_TO_MS if _gs_kts is not None else None)    # → m/s
                    _vs_fpm  = aq.get("VERTICAL_SPEED")     # ft/min
                    slow_cache["vertical_speed"] = (
                        _vs_fpm * 0.00508 if _vs_fpm is not None else None)       # → m/s
                    # BAROMETER_PRESSURE = actual ambient pressure in mbar.
                    # Do NOT use KOHLSMAN_SETTING_MB — that is the QNH dial
                    # setting (often exactly 1013.25), not measured pressure.
                    slow_cache["barometer_pressure"] = aq.get("BAROMETER_PRESSURE")  # mbar
                    slow_cache["latitude"]          = aq.get("PLANE_LATITUDE")   # degrees (GPS GT)
                    slow_cache["longitude"]         = aq.get("PLANE_LONGITUDE")  # degrees (GPS GT)
                    slow_cache["altitude"]          = aq.get("PLANE_ALTITUDE")   # feet
                    slow_cache["pressure_altitude"] = aq.get("PRESSURE_ALTITUDE")  # metres (Python SimConnect returns SI)
                    slow_cache["magnetic_compass"]  = aq.get("MAGNETIC_COMPASS")   # degrees

                row = {
                    "timestamp": time.time(),
                    # accel in ft/s² — step_ekf applies *0.3048 internally
                    "accel_x":  accel_x if accel_x is not None else 0.0,
                    "accel_y":  accel_y if accel_y is not None else 0.0,
                    "accel_z":  accel_z if accel_z is not None else 0.0,
                    "gyro_x":   gyro_x  if gyro_x  is not None else 0.0,
                    "gyro_y":   gyro_y  if gyro_y  is not None else 0.0,
                    "gyro_z":   gyro_z  if gyro_z  is not None else 0.0,
                    "pitch":    pitch   if pitch   is not None else 0.0,
                    "bank":     bank    if bank    is not None else 0.0,
                    **slow_cache,
                }

                # ── Tier 3: time-based frame capture (~5 fps) ─────────────
                # Use wall-clock time, not iteration count, because the loop
                # runs at ~7 Hz (not 50 Hz), so iter_n % 10 would give ~1.4s
                # between frames instead of the intended 0.2s.
                if (t_start - self._last_capture_time) >= capture_interval:
                    img_raw = np.array(
                        self._sct.grab(self._monitor), dtype=np.uint8)
                    img_rgb = img_raw[:, :, 2::-1]  # BGRA → RGB

                    with self._lock:
                        self._latest_row              = row
                        self._latest_img              = img_rgb
                        self._frame_id               += 1
                        self._latest_frame_capture_ts = t_start
                    self._last_capture_time = t_start
                else:
                    with self._lock:
                        self._latest_row = row

            except Exception as exc:
                # Don't crash the background thread on transient errors
                print(f"[SimConnectLiveSource] poll error: {exc}")

            iter_n += 1
            elapsed = time.perf_counter() - t_start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
