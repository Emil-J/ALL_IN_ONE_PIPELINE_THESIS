"""
runtime/simconnect_adapter.py — Data source abstraction for run_pipeline.py.

Two classes:
  FileSource          — reads pre-recorded IMU CSV + frame directory
  SimConnectLiveSource — background-thread SimConnect polling at ~50Hz,
                         non-blocking get_latest_row() / get_latest_frame()

The visual pipeline loop (~1s/frame) calls get_latest_row() / get_latest_frame()
without blocking the background IMU acquisition thread. This keeps the EKF
running at full rate regardless of how long feature matching takes.

SimConnect variable mapping (matches MSFS2020_IMU_Pipeline/data_logger.py):
  ACCELERATION_BODY_X/Y/Z        → accel_x/y/z  (ft/s², converted to m/s²)
  ROTATION_VELOCITY_BODY_X/Y/Z   → gyro_x/y/z   (rad/s, used directly)
  PLANE_HEADING_DEGREES_MAGNETIC → heading_magnetic (radians in SimConnect)
  PLANE_PITCH_DEGREES            → pitch         (rad)
  PLANE_BANK_DEGREES             → bank          (rad)
  KOHLSMAN_SETTING_MB:1          → barometer_pressure (mb)
  AIRSPEED_TRUE                  → airspeed_true  (knots)
  PLANE_LATITUDE / PLANE_LONGITUDE → latitude / longitude (degrees)
"""

import copy
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

# ── ft/s² → m/s² ─────────────────────────────────────────────────────────────
_FT_S2_TO_M_S2 = 0.3048


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

    POLL_HZ           = 50
    CAPTURE_EVERY_N   = 10   # capture frame every 10th poll → ~5fps

    def __init__(self):
        self._lock        = threading.Lock()
        self._latest_row  = None     # dict of IMU data
        self._latest_img  = None     # np.ndarray
        self._frame_id    = 0
        self._running     = False
        self._thread: Optional[threading.Thread] = None

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

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """Return (frame_array_copy_or_None, frame_id)."""
        with self._lock:
            img = self._latest_img
            fid = self._frame_id
        return (img.copy() if img is not None else None), fid

    # ── internal ──────────────────────────────────────────────────────────────

    def _detect_monitor(self) -> dict:
        """Try to find the MSFS window; fall back to primary monitor."""
        try:
            import win32gui
            def _cb(h, acc):
                if win32gui.IsWindowVisible(h):
                    t = win32gui.GetWindowText(h)
                    if "flight simulator" in t.lower():
                        acc.append(h)
            handles = []
            win32gui.EnumWindows(_cb, handles)
            if handles:
                hwnd = handles[0]
                _, _, w, h = win32gui.GetClientRect(hwnd)
                left, top  = win32gui.ClientToScreen(hwnd, (0, 0))
                return {"top": top, "left": left, "width": w, "height": h}
        except Exception:
            pass
        # Fallback — capture full primary screen
        return self._sct.monitors[1]

    def _poll_loop(self):
        aq       = self._aq
        interval = 1.0 / self.POLL_HZ
        iter_n   = 0
        slow_cache = {}
        SLOW_N   = 5   # refresh slow vars every 5 iters

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
                        "PLANE_HEADING_DEGREES_MAGNETIC")
                    slow_cache["airspeed_true"]    = aq.get("AIRSPEED_TRUE")
                    slow_cache["barometer_pressure"] = aq.get(
                        "KOHLSMAN_SETTING_MB:1")
                    slow_cache["latitude"]  = aq.get("PLANE_LATITUDE")
                    slow_cache["longitude"] = aq.get("PLANE_LONGITUDE")

                ts = time.time()
                row = {
                    "timestamp":        ts,
                    "accel_x":  (accel_x or 0.0) * _FT_S2_TO_M_S2,
                    "accel_y":  (accel_y or 0.0) * _FT_S2_TO_M_S2,
                    "accel_z":  (accel_z or 0.0) * _FT_S2_TO_M_S2,
                    "gyro_x":   gyro_x  or 0.0,
                    "gyro_y":   gyro_y  or 0.0,
                    "gyro_z":   gyro_z  or 0.0,
                    "pitch":    pitch   or 0.0,
                    "bank":     bank    or 0.0,
                    **slow_cache,
                }

                # ── Tier 3: frame capture ──────────────────────────────────
                if iter_n % self.CAPTURE_EVERY_N == 0:
                    img_raw = np.array(
                        self._sct.grab(self._monitor), dtype=np.uint8)
                    img_rgb = img_raw[:, :, 2::-1]  # BGRA → RGB

                    with self._lock:
                        self._latest_row = row
                        self._latest_img = img_rgb
                        self._frame_id  += 1
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
