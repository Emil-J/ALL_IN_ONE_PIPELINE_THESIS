"""
Dead Reckoning using IMU data
Integrates accelerometer and gyroscope to estimate position in NED frame
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Earth radius for coordinate conversion
EARTH_RADIUS = 6371000  # meters

# ═══════════════════════════════════════════════════════════════
# QUATERNION UTILITIES
# ═══════════════════════════════════════════════════════════════

def quat_from_euler(roll, pitch, yaw):
    """Convert Euler angles (rad) to quaternion [w, x, y, z]"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quat_to_euler(q):
    """Convert quaternion [w, x, y, z] to Euler angles (rad)"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90° if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def quat_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q):
    """Normalize quaternion to unit length"""
    norm = np.linalg.norm(q)
    if norm > 1e-10:
        return q / norm
    return q

def quat_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix (body to NED)"""
    w, x, y, z = q
    
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle representation"""
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0]*s, axis[1]*s, axis[2]*s])

def ned_to_latlon(north, east, lat0, lon0):
    """Convert NED displacement to lat/lon"""
    # Latitude change
    dlat = north / EARTH_RADIUS
    lat = lat0 + np.degrees(dlat)
    
    # Longitude change (accounting for latitude)
    dlon = east / (EARTH_RADIUS * np.cos(np.radians(lat0)))
    lon = lon0 + np.degrees(dlon)
    
    return lat, lon

def dead_reckoning(csv_path):
    """
    Perform dead reckoning from IMU data
    
    Args:
        csv_path: Path to the flight data CSV from data_logger.py
    
    Returns:
        DataFrame with estimated positions
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required sensors are available
    if 'pressure_altitude' not in df.columns:
        raise ValueError("Barometer (pressure_altitude) required for dead reckoning - GPS altitude not suitable")
    if 'pitch' not in df.columns or 'bank' not in df.columns:
        raise ValueError("Pitch and bank required for gravity synthesis (MSFS accel has no gravity)")
    if 'airspeed_true' not in df.columns:
        print("WARNING: Airspeed sensor not available - velocity estimation will be inaccurate in cruise flight")
    
    # Initialize state
    # Starting position from first GPS reading (initialization only - GPS disabled after takeoff)
    lat0 = df['latitude'].iloc[0]
    lon0 = df['longitude'].iloc[0]
    
    alt0 = df['pressure_altitude'].iloc[0] * 0.3048  # feet to meters
    print(f"Initial position: {lat0:.6f}°N, {lon0:.6f}°E, {alt0:.1f}m (GPS position + barometer altitude)")
    
    # Initialize orientation quaternion from magnetometer heading
    # NOTE: Python SimConnect returns all _DEGREES_ variables in RADIANS
    if 'heading_magnetic' in df.columns and pd.notna(df['heading_magnetic'].iloc[0]):
        heading0_deg = np.degrees(df['heading_magnetic'].iloc[0])
        initial_yaw = np.radians(heading0_deg)
        q = quat_from_euler(0.0, 0.0, initial_yaw)  # Level start, correct heading
        print(f"Initial heading: {heading0_deg:.1f}° from heading_magnetic")
    elif 'magnetic_compass' in df.columns and pd.notna(df['magnetic_compass'].iloc[0]):
        initial_yaw = np.radians(df['magnetic_compass'].iloc[0])
        q = quat_from_euler(0.0, 0.0, initial_yaw)
        print(f"Initial heading: {df['magnetic_compass'].iloc[0]:.1f}° from magnetic_compass (fallback)")
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Fallback: identity quaternion
        print("Warning: No magnetometer data, assuming initial heading = 0°")
    
    # Initialize velocity (m/s) in NED frame
    vel_n = 0.0
    vel_e = 0.0
    vel_d = 0.0
    
    # Initialize position (m) in NED frame relative to start
    pos_n = 0.0
    pos_e = 0.0
    pos_d = 0.0
    
    # Storage for results
    results = []
    
    print("Performing dead reckoning integration...")
    
    for i in range(len(df)):
        # Get time delta
        if i == 0:
            dt = 0
        else:
            dt = df['timestamp'].iloc[i] - df['timestamp'].iloc[i-1]
        
        # Get IMU readings
        # MSFS provides accelerations in ft/s², convert to m/s²
        # MSFS body frame: X=right, Y=up, Z=forward (non-standard!)
        # Standard aviation body frame: X=forward, Y=right, Z=down
        # Remap: MSFS → Standard
        accel_x_msfs = df['accel_x'].iloc[i] * 0.3048  # ft/s² -> m/s²
        accel_y_msfs = df['accel_y'].iloc[i] * 0.3048
        accel_z_msfs = df['accel_z'].iloc[i] * 0.3048
        
        # MSFS accelerometers give coordinate acceleration (no gravity).
        # Real IMUs measure specific force (includes gravity).
        # Synthesize gravity from MSFS pitch/bank so the model works correctly.
        pitch_rad = df['pitch'].iloc[i]   # Python SimConnect returns radians
        bank_rad = df['bank'].iloc[i]     # Python SimConnect returns radians
        g = 9.81
        g_body = np.array([
            -g * np.sin(pitch_rad),
            g * np.sin(bank_rad) * np.cos(pitch_rad),
            g * np.cos(bank_rad) * np.cos(pitch_rad)
        ])
        
        # Remap to standard body frame: [forward, right, down]
        # Linear vectors (acceleration): same-direction axes keep sign,
        # opposite-direction (Y-up→Z-down) gets negated.
        accel_x = accel_z_msfs    # MSFS Z (forward) → Standard X (forward)
        accel_y = accel_x_msfs    # MSFS X (right)   → Standard Y (right)
        accel_z = -accel_y_msfs   # MSFS Y (up)      → Standard Z (down), inverted
        
        # Add synthesized gravity to match real IMU specific force
        accel_x += g_body[0]
        accel_y += g_body[1]
        accel_z += g_body[2]
        
        # Remap gyro rates to match
        # Angular velocity is a pseudovector: handedness change (LH→RH)
        # and axis flip cancel, so no negation needed on any axis.
        gyro_x_msfs = df['gyro_x'].iloc[i]  # rad/s
        gyro_y_msfs = df['gyro_y'].iloc[i]
        gyro_z_msfs = df['gyro_z'].iloc[i]
        
        gyro_x = gyro_z_msfs    # MSFS Z (forward) → Standard X (roll)
        gyro_y = gyro_x_msfs    # MSFS X (right)   → Standard Y (pitch)
        gyro_z = gyro_y_msfs    # MSFS Y (up)      → Standard Z (yaw) — no negation
        
        # ─── SENSOR-AIDED CORRECTIONS ───
        # Use barometer for altitude (much more accurate than integrating accel_z)
        if 'pressure_altitude' in df.columns and pd.notna(df['pressure_altitude'].iloc[i]):
            alt_est = df['pressure_altitude'].iloc[i] * 0.3048  # feet to meters
            pos_d = alt0 - alt_est  # Update down position to match barometer
        
        if dt > 0:
            # ═══ QUATERNION INTEGRATION ═══
            # Integrate gyroscope rates using quaternion derivative
            omega = np.array([gyro_x, gyro_y, gyro_z])  # Body frame angular velocity
            omega_norm = np.linalg.norm(omega)
            
            if omega_norm > 1e-6:
                # Axis-angle integration (more accurate for large rotations)
                angle = omega_norm * dt
                axis = omega / omega_norm
                dq = quat_from_axis_angle(axis, angle)
                q = quat_multiply(q, dq)
            
            # Normalize quaternion to prevent drift
            q = quat_normalize(q)
            
            # ═══ MAGNETOMETER HEADING CORRECTION ═══
            # Apply magnetometer correction directly in quaternion space
            # Prefer heading_magnetic (Python SimConnect returns radians)
            if 'heading_magnetic' in df.columns and pd.notna(df['heading_magnetic'].iloc[i]):
                target_yaw = df['heading_magnetic'].iloc[i]  # Already in radians
            elif 'magnetic_compass' in df.columns and pd.notna(df['magnetic_compass'].iloc[i]):
                target_yaw = np.radians(df['magnetic_compass'].iloc[i])
            else:
                target_yaw = None
            
            if target_yaw is not None:
                
                # Extract current yaw from quaternion (fast method)
                w, x, y, z = q
                current_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                
                # Calculate yaw correction needed
                yaw_error = target_yaw - current_yaw
                
                # Normalize yaw error to [-pi, pi]
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
                
                # Create yaw-only correction quaternion (rotation around Z-axis in NED)
                q_correction = quat_from_axis_angle(np.array([0, 0, 1]), yaw_error)
                
                # Apply correction: q_new = q_correction * q
                q = quat_multiply(q_correction, q)
                q = quat_normalize(q)
            
            # ═══ ACCELERATION TRANSFORMATION ═══
            # Get rotation matrix from quaternion
            R = quat_to_rotation_matrix(q)
            
            # Transform acceleration from body to NED frame
            # accel already includes synthesized gravity (added above),
            # so it behaves like a real IMU specific-force measurement.
            accel_body = np.array([accel_x, accel_y, accel_z])
            accel_ned = R @ accel_body  # Now using correct axis mapping
            
            # Remove gravity (the synthesized gravity was added in body frame;
            # after rotation to NED it appears here and must be subtracted)
            accel_ned[2] -= 9.81
            
            # ═══ VELOCITY FROM AIRSPEED SENSOR ═══
            # Use airspeed sensor for velocity magnitude, heading for direction
            # This is much more accurate for aircraft in cruise flight
            if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[i]):
                airspeed = df['airspeed_true'].iloc[i]  # Already in m/s (data_logger converts from knots)
                
                # Use magnetic heading DIRECTLY for heading (most accurate)
                # Don't use quaternion yaw which may have gyro drift
                if 'heading_magnetic' in df.columns and pd.notna(df['heading_magnetic'].iloc[i]):
                    yaw = df['heading_magnetic'].iloc[i]  # Already in radians
                elif 'magnetic_compass' in df.columns and pd.notna(df['magnetic_compass'].iloc[i]):
                    yaw = np.radians(df['magnetic_compass'].iloc[i])
                else:
                    # Fallback: use quaternion yaw
                    _, _, yaw = quat_to_euler(q)
                
                # Convert heading to velocity components
                # vel_n = V * cos(yaw), vel_e = V * sin(yaw)
                vel_n = airspeed * np.cos(yaw)
                vel_e = airspeed * np.sin(yaw)
            else:
                # Fallback: integrate acceleration (less accurate in cruise)
                vel_n += accel_ned[0] * dt
                vel_e += accel_ned[1] * dt
            
            # Vertical velocity from barometer changes (more accurate)
            if i > 0 and 'pressure_altitude' in df.columns:
                alt_prev = df['pressure_altitude'].iloc[i-1] * 0.3048
                alt_curr = df['pressure_altitude'].iloc[i] * 0.3048
                vel_d = -(alt_curr - alt_prev) / dt  # Down velocity (negative = climbing)
            else:
                vel_d += accel_ned[2] * dt
            
            # ═══ POSITION INTEGRATION ═══
            # Integrate velocity to get position (horizontal only, altitude from barometer)
            pos_n += vel_n * dt
            pos_e += vel_e * dt
            # pos_d already updated from barometer above
        
        # Convert NED position to lat/lon
        lat_est, lon_est = ned_to_latlon(pos_n, pos_e, lat0, lon0)
        if 'pressure_altitude' not in df.columns:
            alt_est = alt0 - pos_d  # Fallback to integrated altitude
        # else alt_est already set from barometer above
        
        # Convert quaternion to Euler angles for output/debugging
        roll, pitch, yaw = quat_to_euler(q)
        
        # Store results
        results.append({
            'timestamp': df['timestamp'].iloc[i],
            'latitude_est': lat_est,
            'longitude_est': lon_est,
            'altitude_est': alt_est,
            'pos_n': pos_n,
            'pos_e': pos_e,
            'pos_d': pos_d,
            'vel_n': vel_n,
            'vel_e': vel_e,
            'vel_d': vel_d,
            'roll': np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw)
        })
    
    return pd.DataFrame(results)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Dead reckoning from IMU data')
    parser.add_argument('--input', type=str, help='Path to flight data CSV')
    parser.add_argument('--output', type=str, help='Path to save dead reckoning results CSV')
    args = parser.parse_args()
    
    log_dir = "logs"
    
    if args.input:
        csv_path = args.input
    else:
        # Find the most recent flight data file
        if not os.path.exists(log_dir):
            print("Error: logs directory not found. Run data_logger.py first.")
            return
        
        csv_files = [f for f in os.listdir(log_dir) if f.startswith("imu_gps_log_") and f.endswith(".csv")]
        if not csv_files:
            print("Error: No flight data CSV found. Run data_logger.py first.")
            return
        
        # Get most recent
        csv_files.sort(reverse=True)
        csv_path = os.path.join(log_dir, csv_files[0])
        print(f"Using most recent data file: {csv_files[0]}")
    
    # Run dead reckoning
    results_df = dead_reckoning(csv_path)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"dead_reckoning_{timestamp_str}.csv"
        output_path = os.path.join("logs", output_filename)
    
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n✓ Dead reckoning complete!")
    print(f"✓ Processed {len(results_df)} data points")
    print(f"✓ Results saved to: {os.path.basename(output_path)}")
    
    # Show final drift
    initial_lat = results_df['latitude_est'].iloc[0]
    initial_lon = results_df['longitude_est'].iloc[0]
    final_lat = results_df['latitude_est'].iloc[-1]
    final_lon = results_df['longitude_est'].iloc[-1]
    
    print(f"\nInitial estimated position: {initial_lat:.6f}°N, {initial_lon:.6f}°E")
    print(f"Final estimated position: {final_lat:.6f}°N, {final_lon:.6f}°E")
    print(f"Total displacement: N={results_df['pos_n'].iloc[-1]:.1f}m, E={results_df['pos_e'].iloc[-1]:.1f}m")
    
    return output_path

if __name__ == "__main__":
    main()
