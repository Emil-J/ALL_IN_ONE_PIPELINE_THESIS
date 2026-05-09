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

# Configuration
IMU_RATE_HZ = 50  # ~50Hz for IMU + GPS logging
CAMERA_RATE_HZ = 5  # 5fps for image capture
LOG_DIR = "logs"

# Screen capture region for Monitor 2 (1920x1080 at left=-1920, top=51)
# Captures the main MSFS viewport, excluding bottom instrument panel
CAPTURE_REGION = {"top": 100, "left": -1920, "width": 1920, "height": 700}

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
    aq = AircraftRequests(sm, _time=0)
    ae = AircraftEvents(sm)
    
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
    
    # Initialize data storage
    data_log = []
    
    # Initialize screen capture
    sct = mss.mss()
    monitor = CAPTURE_REGION
    
    # Timing control
    imu_interval = 1.0 / IMU_RATE_HZ
    camera_interval = 1.0 / CAMERA_RATE_HZ
    
    start_time = time.time()
    last_imu_time = start_time
    last_camera_time = start_time
    frame_count = 0
    
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
                # Read IMU data
                accel_x = aq.get("ACCELERATION_BODY_X")
                accel_y = aq.get("ACCELERATION_BODY_Y")
                accel_z = aq.get("ACCELERATION_BODY_Z")
                
                gyro_x = aq.get("ROTATION_VELOCITY_BODY_X")
                gyro_y = aq.get("ROTATION_VELOCITY_BODY_Y")
                gyro_z = aq.get("ROTATION_VELOCITY_BODY_Z")
                
                # Read GPS data (for ground truth only - not used in dead reckoning)
                latitude = aq.get("PLANE_LATITUDE")
                longitude = aq.get("PLANE_LONGITUDE")
                altitude = aq.get("PLANE_ALTITUDE")
                
                # Read attitude data for debugging
                pitch = aq.get("PLANE_PITCH_DEGREES")
                bank = aq.get("PLANE_BANK_DEGREES")
                heading = aq.get("PLANE_HEADING_DEGREES_TRUE")
                
                # ─── ADDITIONAL SENSORS FOR DEAD RECKONING ───
                # Barometer - pressure altitude (independent of GPS)
                pressure_altitude = aq.get("PRESSURE_ALTITUDE")  # meters Standard Altitude, ie: at a 1013.25 hPa (1 atmosphere) setting.
                barometer_pressure = aq.get("BAROMETER_PRESSURE")  # millibars
                
                # Magnetometer - magnetic heading (independent of GPS)
                heading_magnetic = aq.get("PLANE_HEADING_DEGREES_MAGNETIC")  # degrees
                magnetic_compass = aq.get("MAGNETIC_COMPASS")  # compass reading
                
                # Additional useful sensors (MSFS returns these in non-SI units)
                airspeed_indicated_kts = aq.get("AIRSPEED_INDICATED")  # knots
                airspeed_true_kts = aq.get("AIRSPEED_TRUE")  # knots
                ground_velocity_kts = aq.get("GROUND_VELOCITY")  # knots
                vertical_speed_fpm = aq.get("VERTICAL_SPEED")  # feet per minute
                
                # Convert to SI units (m/s) for storage
                airspeed_indicated = airspeed_indicated_kts * 0.514444 if airspeed_indicated_kts else None
                airspeed_true = airspeed_true_kts * 0.514444 if airspeed_true_kts else None
                ground_velocity = ground_velocity_kts * 0.514444 if ground_velocity_kts else None
                vertical_speed = vertical_speed_fpm * 0.00508 if vertical_speed_fpm else None
                
                # Read autopilot states for debugging
                ap_master = aq.get("AUTOPILOT_MASTER")
                ap_alt_hold = aq.get("AUTOPILOT_ALTITUDE_LOCK")
                ap_airspeed_hold = aq.get("AUTOPILOT_AIRSPEED_HOLD")
                ap_nav_hold = aq.get("AUTOPILOT_NAV1_LOCK")
                throttle_pos = aq.get("GENERAL_ENG_THROTTLE_LEVER_POSITION:1")
                
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
                
                # Print detailed status every second
                if len(data_log) % IMU_RATE_HZ == 0:
                    print(f"\n[{elapsed:.1f}s] Samples: {len(data_log)} | Frames: {frame_count}")
                    print(f"  Position: {latitude:.6f}°N, {longitude:.6f}°E, {altitude:.1f}ft")
                    print(f"  Attitude: Pitch {pitch:.1f}°, Bank {bank:.1f}°, Heading {heading:.1f}°")
                    # Display in aviation-friendly units (knots/fpm) even though stored in m/s
                    print(f"  Speed: IAS {airspeed_indicated_kts:.1f}kts, TAS {airspeed_true_kts:.1f}kts, GS {ground_velocity_kts:.1f}kts")
                    print(f"  Vertical: {vertical_speed_fpm:.0f} ft/min, Throttle: {throttle_pos:.0f}%")
                    print(f"  Autopilot: Master={ap_master:.0f}, Alt={ap_alt_hold:.0f}, Airspeed={ap_airspeed_hold:.0f}, Nav={ap_nav_hold:.0f}")
            
            # Camera capture at 2-5fps
            if current_time - last_camera_time >= camera_interval:
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
