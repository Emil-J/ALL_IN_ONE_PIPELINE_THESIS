"""
MSFS 2020 Data Logger
Captures IMU data (accelerometer + gyroscope), GPS, and nadir camera frames

IMPORTANT - MSFS SimConnect Units:
- Accelerations: ft/s² (converted to m/s²)
- Angular rates: rad/s (used directly)
- Altitudes: feet (converted to meters where needed)
- Airspeeds: knots (converted to m/s for storage)
- All velocity data stored in CSV is in SI units (m/s)
"""

from SimConnect import *
import mss
import numpy as np
import pandas as pd
import time
import os
import msvcrt
from datetime import datetime
from PIL import Image
import win32gui

# Configuration
IMU_RATE_HZ = 50  # ~50Hz for IMU + GPS logging
CAMERA_RATE_HZ = 5  # 5fps for image capture
LOG_DIR = "logs"

# Two-tier polling: each aq.get() takes ~18ms roundtrip to SimConnect.
# With 25 vars × 18ms = 450ms per loop = 2.2Hz — FAR below the 50Hz target.
# Solution: only poll critical IMU+attitude vars every iteration (8 vars → ~144ms → 7Hz),
# poll everything else every SLOW_POLL_INTERVAL iterations and forward-fill between.
SLOW_POLL_INTERVAL = 5  # Poll non-critical vars every 5th iteration

# ─── SCREEN CAPTURE CONFIGURATION ───
# MSFS window detection: automatically finds the game window
# Set MSFS_WINDOW_TITLE to match your game window title
MSFS_WINDOW_TITLE = "Microsoft Flight Simulator"

# Fallback capture region if MSFS window cannot be detected
# (e.g., exclusive fullscreen mode). Sized for a 2560x1440 monitor.
FALLBACK_CAPTURE_REGION = {"top": 0, "left": 0, "width": 2560, "height": 1440}


def find_msfs_window():
    """Find the MSFS 2020 window handle by title.
    Returns the window handle (hwnd) or None if not found."""
    # Try exact title match first
    hwnd = win32gui.FindWindow(None, MSFS_WINDOW_TITLE)
    if hwnd and win32gui.IsWindowVisible(hwnd):
        return hwnd

    # Fallback: partial match (handles title variations)
    results = []
    def _enum_cb(h, acc):
        if win32gui.IsWindowVisible(h):
            title = win32gui.GetWindowText(h)
            if "flight simulator" in title.lower():
                acc.append(h)
    win32gui.EnumWindows(_enum_cb, results)
    return results[0] if results else None


def get_msfs_capture_region(hwnd):
    """Get the mss-compatible capture region for the MSFS game viewport.
    Uses GetClientRect to capture only the rendered game content,
    excluding the title bar, window borders, and shadow."""
    # GetClientRect gives (0, 0, width, height) of the client area
    _, _, width, height = win32gui.GetClientRect(hwnd)
    # ClientToScreen converts client (0,0) to absolute screen coordinates
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return {"top": top, "left": left, "width": width, "height": height}

def setup_directories(timestamp_str):
    """Create log directories for this run"""
    os.makedirs(LOG_DIR, exist_ok=True)
    images_dir = os.path.join(LOG_DIR, f"images_{timestamp_str}")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir

def main():
    # Create timestamp for this run
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = setup_directories(timestamp_str)
    
    # Connect to MSFS
    print("Connecting to Microsoft Flight Simulator 2020...")
    sm = SimConnect()
    # _time=20: cache each SimConnect variable for 20ms (50Hz refresh).
    # With _time=0 every aq.get() would block for a fresh roundtrip (~18ms each).
    # 25 calls × 18ms = 450ms per loop = 2.2 Hz instead of the intended 50 Hz.
    aq = AircraftRequests(sm, _time=20)
    
    print("Connected!")
    
    # ─── MANUAL FLIGHT MODE ───
    print("\n" + "="*60)
    print("READY TO LOG DATA")
    print("="*60)
    print("✓ SimConnect connected")
    print("✓ Waiting for you to start/control the flight in MSFS")
    print("✓ Use MSFS autopilot, AI pilot, or fly manually")
    print("="*60 + "\n")
    
    print("Starting data collection...")
    print(f"IMU Rate: {IMU_RATE_HZ}Hz, Camera Rate: {CAMERA_RATE_HZ}fps")
    print(f"Run ID: {timestamp_str}")
    print("Press 'q' and Enter to stop logging\n")
    # ─── WAIT FOR VALID GPS ───
    # SimConnect returns ~0,0 coordinates while the aircraft is still loading.
    # We wait until we get a real position before starting the clock.
    print("Waiting for valid GPS position from MSFS...")
    while True:
        lat_check = aq.get("PLANE_LATITUDE")
        lon_check = aq.get("PLANE_LONGITUDE")
        if lat_check is not None and lon_check is not None and (abs(lat_check) > 1.0 or abs(lon_check) > 1.0):
            print(f"✓ GPS ready: {lat_check:.5f}°N, {lon_check:.5f}°E")
            break
        print(f"  GPS not ready yet ({lat_check}, {lon_check}) - waiting...")
        time.sleep(0.5)
    
    # Initialize data storage
    data_log = []
    
    # Initialize screen capture
    sct = mss.mss()
    
    # ─── DETECT MSFS WINDOW ───
    msfs_hwnd = find_msfs_window()
    if msfs_hwnd:
        monitor = get_msfs_capture_region(msfs_hwnd)
        title = win32gui.GetWindowText(msfs_hwnd)
        print(f"✓ MSFS window found: \"{title}\"")
        print(f"  Capture region: {monitor}")
    else:
        monitor = FALLBACK_CAPTURE_REGION
        print("⚠ MSFS window not detected (exclusive fullscreen?)")
        print(f"  Using fallback region: {monitor}")
    
    # Timing control
    imu_interval = 1.0 / IMU_RATE_HZ
    camera_interval = 1.0 / CAMERA_RATE_HZ
    
    start_time = time.time()
    last_imu_time = start_time
    last_camera_time = start_time
    frame_count = 0
    
    # Forward-fill storage for slow-polled variables (Tier 2)
    slow_vars = {
        'heading_magnetic': None, 'airspeed_true_kts': None,
        'barometer_pressure': None, 'latitude': None, 'longitude': None,
        'altitude': None, 'heading': None, 'pressure_altitude': None,
        'magnetic_compass': None, 'airspeed_indicated_kts': None,
        'ground_velocity_kts': None, 'vertical_speed_fpm': None,
        'ap_master': None, 'ap_alt_hold': None, 'ap_airspeed_hold': None,
        'ap_nav_hold': None, 'throttle_pos': None,
    }
    
    stop_logging = False
    
    try:
        while not stop_logging:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if user pressed 'q' to stop
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'q':
                    print("\n'q' pressed - stopping logging...")
                    stop_logging = True
                    break
            
            # IMU + GPS logging at ~50Hz
            if current_time - last_imu_time >= imu_interval:
                # ─── TIER 1: CRITICAL VARS (every iteration) ───
                # 8 vars × ~18ms = ~144ms → ~7Hz
                # These are the minimum for EKF predict + gravity synthesis
                accel_x = aq.get("ACCELERATION_BODY_X")
                accel_y = aq.get("ACCELERATION_BODY_Y")
                accel_z = aq.get("ACCELERATION_BODY_Z")
                
                gyro_x = aq.get("ROTATION_VELOCITY_BODY_X")
                gyro_y = aq.get("ROTATION_VELOCITY_BODY_Y")
                gyro_z = aq.get("ROTATION_VELOCITY_BODY_Z")
                
                pitch = aq.get("PLANE_PITCH_DEGREES")   # needed for gravity synthesis
                bank = aq.get("PLANE_BANK_DEGREES")     # needed for gravity synthesis
                
                # ─── TIER 2: SLOW VARS (every Nth iteration) ───
                # Heading, airspeed, baro, GPS, debug — polled less often, forward-filled
                if len(data_log) % SLOW_POLL_INTERVAL == 0:
                    heading_magnetic = aq.get("PLANE_HEADING_DEGREES_MAGNETIC")
                    airspeed_true_kts = aq.get("AIRSPEED_TRUE")
                    barometer_pressure = aq.get("BAROMETER_PRESSURE")
                    latitude = aq.get("PLANE_LATITUDE")
                    longitude = aq.get("PLANE_LONGITUDE")
                    altitude = aq.get("PLANE_ALTITUDE")
                    heading = aq.get("PLANE_HEADING_DEGREES_TRUE")
                    pressure_altitude = aq.get("PRESSURE_ALTITUDE")
                    magnetic_compass = aq.get("MAGNETIC_COMPASS")
                    airspeed_indicated_kts = aq.get("AIRSPEED_INDICATED")
                    ground_velocity_kts = aq.get("GROUND_VELOCITY")
                    vertical_speed_fpm = aq.get("VERTICAL_SPEED")
                    ap_master = aq.get("AUTOPILOT_MASTER")
                    ap_alt_hold = aq.get("AUTOPILOT_ALTITUDE_LOCK")
                    ap_airspeed_hold = aq.get("AUTOPILOT_AIRSPEED_HOLD")
                    ap_nav_hold = aq.get("AUTOPILOT_NAV1_LOCK")
                    throttle_pos = aq.get("GENERAL_ENG_THROTTLE_LEVER_POSITION:1")
                    
                    # Update last-known slow values
                    slow_vars['heading_magnetic'] = heading_magnetic
                    slow_vars['airspeed_true_kts'] = airspeed_true_kts
                    slow_vars['barometer_pressure'] = barometer_pressure
                    slow_vars['latitude'] = latitude
                    slow_vars['longitude'] = longitude
                    slow_vars['altitude'] = altitude
                    slow_vars['heading'] = heading
                    slow_vars['pressure_altitude'] = pressure_altitude
                    slow_vars['magnetic_compass'] = magnetic_compass
                    slow_vars['airspeed_indicated_kts'] = airspeed_indicated_kts
                    slow_vars['ground_velocity_kts'] = ground_velocity_kts
                    slow_vars['vertical_speed_fpm'] = vertical_speed_fpm
                    slow_vars['ap_master'] = ap_master
                    slow_vars['ap_alt_hold'] = ap_alt_hold
                    slow_vars['ap_airspeed_hold'] = ap_airspeed_hold
                    slow_vars['ap_nav_hold'] = ap_nav_hold
                    slow_vars['throttle_pos'] = throttle_pos
                else:
                    # Use forward-filled values from last slow poll
                    heading_magnetic = slow_vars['heading_magnetic']
                    airspeed_true_kts = slow_vars['airspeed_true_kts']
                    barometer_pressure = slow_vars['barometer_pressure']
                    latitude = slow_vars['latitude']
                    longitude = slow_vars['longitude']
                    altitude = slow_vars['altitude']
                    heading = slow_vars['heading']
                    pressure_altitude = slow_vars['pressure_altitude']
                    magnetic_compass = slow_vars['magnetic_compass']
                    airspeed_indicated_kts = slow_vars['airspeed_indicated_kts']
                    ground_velocity_kts = slow_vars['ground_velocity_kts']
                    vertical_speed_fpm = slow_vars['vertical_speed_fpm']
                    ap_master = slow_vars['ap_master']
                    ap_alt_hold = slow_vars['ap_alt_hold']
                    ap_airspeed_hold = slow_vars['ap_airspeed_hold']
                    ap_nav_hold = slow_vars['ap_nav_hold']
                    throttle_pos = slow_vars['throttle_pos']
                
                # Convert to SI units (m/s) for storage
                airspeed_indicated = airspeed_indicated_kts * 0.514444 if airspeed_indicated_kts else None
                airspeed_true = airspeed_true_kts * 0.514444 if airspeed_true_kts else None
                ground_velocity = ground_velocity_kts * 0.514444 if ground_velocity_kts else None
                vertical_speed = vertical_speed_fpm * 0.00508 if vertical_speed_fpm else None
                
                # Store data point
                data_log.append({
                    'timestamp': elapsed,
                    'accel_x': accel_x,
                    'accel_y': accel_y,
                    'accel_z': accel_z,
                    'gyro_x': gyro_x,
                    'gyro_y': gyro_y,
                    'gyro_z': gyro_z,
                    # Ground truth (GPS) - for evaluation only
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude,
                    # Attitude
                    'pitch': pitch,
                    'bank': bank,
                    'heading': heading,
                    # Additional sensors for dead reckoning
                    'pressure_altitude': pressure_altitude,
                    'barometer_pressure': barometer_pressure,
                    'heading_magnetic': heading_magnetic,
                    'magnetic_compass': magnetic_compass,
                    # Velocity
                    'airspeed_indicated': airspeed_indicated,
                    'airspeed_true': airspeed_true,
                    'ground_velocity': ground_velocity,
                    'vertical_speed': vertical_speed,
                    # Debug
                    'ap_master': ap_master,
                    'ap_alt_hold': ap_alt_hold,
                    'ap_airspeed_hold': ap_airspeed_hold,
                    'ap_nav_hold': ap_nav_hold,
                    'throttle_pos': throttle_pos
                })
                
                last_imu_time = current_time
                
                # Print status every second - show actual achieved Hz
                if len(data_log) % IMU_RATE_HZ == 0:
                    actual_hz = len(data_log) / elapsed if elapsed > 0 else 0
                    print(f"\n[{elapsed:.1f}s] Samples: {len(data_log)} | Frames: {frame_count} | Rate: {actual_hz:.1f} Hz (target {IMU_RATE_HZ} Hz)")
                    print(f"  Position: {latitude:.6f}°N, {longitude:.6f}°E, {altitude:.1f}ft")
                    print(f"  Attitude: Pitch {pitch:.1f}°, Bank {bank:.1f}°, Heading {heading:.1f}°")
                    # Display in aviation-friendly units (knots/fpm) even though stored in m/s
                    print(f"  Speed: IAS {airspeed_indicated_kts:.1f}kts, TAS {airspeed_true_kts:.1f}kts, GS {ground_velocity_kts:.1f}kts")
                    print(f"  Vertical: {vertical_speed_fpm:.0f} ft/min, Throttle: {throttle_pos:.0f}%")
                    print(f"  Autopilot: Master={ap_master:.0f}, Alt={ap_alt_hold:.0f}, Airspeed={ap_airspeed_hold:.0f}, Nav={ap_nav_hold:.0f}")
            
            # Camera capture at 2-5fps
            if current_time - last_camera_time >= camera_interval:
                # Re-detect MSFS window position (handles window moves/resizes)
                if msfs_hwnd:
                    try:
                        monitor = get_msfs_capture_region(msfs_hwnd)
                    except Exception:
                        pass  # Keep last known region if detection fails momentarily
                
                # Capture screen
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                # Save with timestamp
                frame_filename = f"frame_{elapsed:.3f}.jpg"
                frame_path = os.path.join(images_dir, frame_filename)
                img.save(frame_path, quality=85)
                
                frame_count += 1
                last_camera_time = current_time
            
            # Small sleep to avoid CPU spinning
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed - stopping data collection...")
    except Exception as e:
        print(f"\n\nError during logging: {e}")
    
    # Save CSV
    df = pd.DataFrame(data_log)
    csv_filename = f"imu_gps_log_{timestamp_str}.csv"
    csv_path = os.path.join(LOG_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Saved {len(data_log)} data points to: {csv_filename}")
    print(f"✓ Saved {frame_count} frames to: images_{timestamp_str}/")
    print(f"✓ Total flight time: {elapsed:.1f}s")
    print(f"\nData saved to: {os.path.abspath(LOG_DIR)}")

if __name__ == "__main__":
    main()
