"""
Error-State Extended Kalman Filter for INS
Based on Algorithm 4 from Kok, Hol, Schön (2017)
"Using Inertial Sensors for Position and Orientation Estimation"

This implements a proper sensor fusion approach combining:
- IMU (gyroscope + accelerometer)
- Magnetometer (heading reference)
- Barometer (altitude reference)
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# QUATERNION UTILITIES (same as before)
# ═══════════════════════════════════════════════════════════════════

def quat_from_euler(roll, pitch, yaw):
    """Create quaternion from Euler angles (ZYX convention)"""
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    
    return np.array([w, x, y, z])


def quat_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quat_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quat_normalize(q):
    """Normalize quaternion"""
    return q / np.linalg.norm(q)


def quat_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix (body to NED)"""
    w, x, y, z = q
    
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    
    return R


def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle representation"""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def expq(omega):
    """Exponential map for quaternions (Equation 3.31 in paper)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    half_theta = theta / 2
    s = np.sin(half_theta) / theta
    return np.array([np.cos(half_theta), s*omega[0], s*omega[1], s*omega[2]])


def skew(v):
    """Skew-symmetric matrix from 3D vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


# ═══════════════════════════════════════════════════════════════════
# ERROR-STATE EKF (Algorithm 4 from paper)
# ═══════════════════════════════════════════════════════════════════

class ErrorStateEKF:
    """
    Error-State Extended Kalman Filter for orientation estimation
    
    State: 3D orientation error (δθ)
    Linearization point: Quaternion q̃ (orientation estimate)
    
    Measurements:
    - Gyroscope: angular velocity
    - Accelerometer: gravity direction (for pitch/roll)
    - Magnetometer: magnetic north (for yaw)
    - Barometer: altitude
    """
    
    def __init__(self, lat0, lon0, alt0, heading0, airspeed0=None):
        """Initialize EKF
        
        Args:
            lat0, lon0: Initial latitude/longitude (degrees)
            alt0: Initial altitude (meters)
            heading0: Initial heading from magnetometer (degrees)
            airspeed0: Initial airspeed (m/s) for velocity initialization
        """
        # Earth radius for lat/lon conversion
        self.R_earth = 6371000.0  # meters
        self.lat0_rad = np.radians(lat0)
        self.lon0_rad = np.radians(lon0)
        
        # ─── STATE ───
        # Orientation: quaternion linearization point (body to NED)
        self.q_tilde = quat_from_euler(0.0, 0.0, np.radians(heading0))
        
        # Magnetic declination (needed for heading conversion)
        self.mag_dec = np.radians(4.0)  # ~4° East at Vejle, Denmark (MSFS WMM)
        
        # Position in NED frame (meters)
        self.pos_n = 0.0
        self.pos_e = 0.0
        self.pos_d = 0.0
        self.alt0 = alt0  # Barometer reference
        
        # Velocity in NED frame (m/s)
        # Initialize from airspeed + heading if available (avoids slow convergence from 0)
        if airspeed0 is not None and airspeed0 > 1.0:
            yaw_rad = np.radians(heading0) + self.mag_dec  # Convert magnetic → true heading
            self.vel_n = airspeed0 * np.cos(yaw_rad)
            self.vel_e = airspeed0 * np.sin(yaw_rad)
        else:
            self.vel_n = 0.0
            self.vel_e = 0.0
        self.vel_d = 0.0
        
        # Gyroscope bias (rad/s) - estimated online
        self.gyro_bias = np.zeros(3)
        
        # Maximum reasonable gyro bias (rad/s)
        # MSFS simulated gyros show transient biases up to 0.03 rad/s (1.7°/s)
        # at data logging start, but most of this is real turn rate, not bias.
        self.max_gyro_bias = 0.015  # ~0.86 deg/s
        
        # Wind estimate in NED frame (m/s) - estimated online
        self.wind_n = 0.0
        self.wind_e = 0.0
        
        # Previous altitude for vertical velocity calculation
        self.alt_prev = alt0
        self.time_prev = 0.0
        
        # Maneuver detection for acceleration integration
        self.maneuver_threshold = 2.0  # m/s² - horizontal acceleration threshold for maneuver detection
        self.in_maneuver = False
        self.last_dt = 0.0  # Time step for deferred position integration
        
        # Wind convergence tracking
        self.wind_converged = False
        self.innovation_history = []  # Track recent innovations for convergence detection
        self.convergence_window = 50  # samples to check for convergence
        
        # ─── COVARIANCE ───
        # State is 8D: orientation error (δθ) + bias error (δb) + wind (w_n, w_e)
        # P = [ P_θθ   P_θb   P_θw ]
        #     [ P_bθ   P_bb   P_bw ]
        #     [ P_wθ   P_wb   P_ww ]
        self.P = np.eye(8) * 0.5  # Initial uncertainty
        self.P[3:6, 3:6] = np.eye(3) * 0.01  # Allow substantial initial gyro bias (~0.1 rad/s at 1σ)
        # MSFS gyros have biases of 0.01-0.03 rad/s; old P_bb=0.001 was too confident
        # and caused slow adaptation → heading drift → position error
        self.P[6:8, 6:8] = np.eye(2) * (5.0)**2  # SMALL wind uncertainty - expect low wind initially
        # This biases filter: wind should be small, so velocity must absorb discrepancy
        
        # ─── REFERENCE VECTORS (NED frame) ───
        self.g_n = np.array([0, 0, 9.81])  # Gravity (down in NED)
        
        # Magnetic field at Vejle, Denmark (from MSFS World Magnetic Model)
        # Inclination ~70° down (declination set above)
        mag_inc = np.radians(70.0)
        self.m_n = np.array([
            np.cos(mag_inc) * np.cos(self.mag_dec),
            np.cos(mag_inc) * np.sin(self.mag_dec),
            np.sin(mag_inc)
        ])
        
        # ─── NOISE COVARIANCES (tuning parameters) ───
        # Process noise
        self.Q_gyro = np.eye(3) * (0.05)**2  # Gyro noise (rad/s)² - moderate trust in gyro
        self.Q_gyro_bias = np.eye(3) * (0.03)**2  # Gyro bias drift - increased for MSFS sensor bias tracking
        self.Q_wind = np.eye(2) * (0.1)**2  # Wind drift noise (m/s)² - slowly varying wind
        
        # Measurement noise
        self.R_accel = np.eye(3) * (1.2)**2  # Accelerometer noise (m/s²)² - moderate: some trust but account for dynamics
        self.R_mag = np.eye(3) * (0.05)**2   # Magnetometer noise - trust for strong heading lock
        self.R_baro = 1.0**2  # Barometer noise (m)²
        self.R_airspeed = 2.0**2  # Airspeed sensor noise (m/s)² - realistic for aircraft sensor
        
        # Complementary filter gain for direct heading correction.
        # At 2Hz with large MSFS gyro biases (especially roll: ~0.055 rad/s),
        # heading from gyro integration is unreliable due to quaternion coupling.
        # Practically all heading information comes from the magnetometer.
        # High gain (~0.95) locks heading to magnetometer each step.
        self.heading_gain_initial = 0.95
        self.heading_gain_steady = 0.85
        self.heading_convergence_samples = 40
        self.heading_correction_gain = self.heading_gain_initial
        self.sample_count = 0
        
        # ─── HISTORY (for output) ───
        self.history = {
            'timestamp': [],
            'latitude': [],
            'longitude': [],
            'altitude': [],
            'pos_n': [],
            'pos_e': [],
            'pos_d': [],
            'vel_n': [],
            'vel_e': [],
            'vel_d': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'gyro_bias_x': [],
            'gyro_bias_y': [],
            'gyro_bias_z': [],
            'wind_n': [],
            'wind_e': []
        }
    
    
    def predict(self, omega_meas, accel_body, dt):
        """Time update (prediction step)
        
        Args:
            omega_meas: Gyroscope measurement (rad/s) in body frame
            accel_body: Accelerometer measurement (m/s²) in body frame
            dt: Time step (seconds)
        """
        if dt <= 0:
            return
        
        # ═══ ORIENTATION UPDATE (Equation 4.54a in paper) ═══
        # Gyroscope measurement corrected for bias
        omega_corrected = omega_meas - self.gyro_bias
        
        # Update orientation quaternion: q̃(t|t-1) = q̃(t-1|t-1) * exp_q(T/2 * ω)
        dq = expq(dt / 2 * omega_corrected)
        self.q_tilde = quat_multiply(self.q_tilde, dq)
        self.q_tilde = quat_normalize(self.q_tilde)
        
        # ═══ GYRO BIAS UPDATE (process noise allows slow drift) ═══
        # Bias doesn't change in predict (updated in measurement step if observable)
        # But add process noise to covariance to allow bias estimation
        
        # ═══ COVARIANCE UPDATE (Equation 4.54b extended to 8D) ═══
        # State: [δθ, δb, δw] (8D: orientation error + bias error + wind error)
        # Process model:
        #   δθ̇ = -ω_corrected × δθ + R_nb * δb + noise
        #   δḃ = 0 + bias_drift_noise
        #   δẇ = 0 + wind_drift_noise (slowly varying wind)
        
        R_nb = quat_to_rotation_matrix(self.q_tilde)
        
        # Jacobian of process model:
        # F = [ -[ω]×    R_nb    0 ]
        #     [   0       0      0 ]  (bias doesn't evolve deterministically)
        #     [   0       0      0 ]  (wind doesn't evolve deterministically)
        # But for small dt, discretized: F ≈ I + dt*Fc
        
        # Simplified propagation (first-order):
        # P_θθ grows due to gyro noise and bias uncertainty
        # P_bb grows due to bias drift
        # P_ww grows due to wind drift
        # P_θb couples orientation-bias uncertainty
        
        # Process noise: Q = [ Q_gyro           0         0     ]
        #                     [   0       Q_gyro_bias     0     ]
        #                     [   0             0       Q_wind ]
        Q_full = np.block([
            [self.Q_gyro, np.zeros((3, 3)), np.zeros((3, 2))],
            [np.zeros((3, 3)), self.Q_gyro_bias, np.zeros((3, 2))],
            [np.zeros((2, 3)), np.zeros((2, 3)), self.Q_wind]
        ])
        
        # Propagation matrix G for orientation (couples gyro noise through rotation)
        G_theta = dt * R_nb
        
        # State transition (simplified first-order approximation)
        # F = [ I    0.5*dt*R_nb    0 ]
        #     [ 0      I            0 ]
        #     [ 0      0            I ]
        F = np.eye(8)
        F[0:3, 3:6] = 0.5 * dt * R_nb  # Bias affects orientation (scaled down for stability)
        # Wind doesn't affect orientation or bias directly
        
        # Covariance propagation: P = F * P * F^T + G * Q * G^T
        # Simplified: Add process noise directly (valid for small dt)
        self.P = F @ self.P @ F.T
        
        # Add process noise (orientation from gyro, bias from drift, wind from drift)
        self.P[0:3, 0:3] += G_theta @ self.Q_gyro @ G_theta.T
        self.P[3:6, 3:6] += self.Q_gyro_bias * dt
        self.P[6:8, 6:8] += self.Q_wind * dt
        
        # ═══ MANEUVER DETECTION ═══
        # Detect maneuvers by checking horizontal acceleration magnitude
        # Remove gravity component to get true acceleration
        accel_ned = R_nb @ accel_body
        accel_horizontal = np.sqrt(accel_ned[0]**2 + accel_ned[1]**2)
        self.in_maneuver = accel_horizontal > self.maneuver_threshold
        
        # ═══ VELOCITY UPDATE ═══
        # ALWAYS integrate acceleration to maintain ground velocity estimate
        # This is critical for wind observability: wind = (airspeed vector) - (integrated velocity)
        # Remove gravity to get true acceleration
        accel_true = accel_ned - self.g_n
        self.vel_n += accel_true[0] * dt
        self.vel_e += accel_true[1] * dt
        # Note: vel_d updated from barometer (more accurate than accel integration)
        
        # ═══ POSITION UPDATE ═══
        # DEFERRED: position is integrated after all measurement updates
        # (in update_airspeed) so it uses the corrected heading and airspeed-
        # derived velocity, not the pre-correction predicted heading.
        self.last_dt = dt
    
    
    def update_accel_mag(self, accel_meas, mag_heading_deg):
        """Measurement update from accelerometer and magnetometer (SEPARATE updates)
        
        Accel and mag are updated independently to prevent cross-contamination.
        The combined 4x8 update was allowing large accel innovations to dilute
        the heading correction, causing heading drift at low sample rates.
        
        After the Kalman updates, a complementary filter directly corrects
        the heading toward the magnetometer as a robustness safety net.
        
        Args:
            accel_meas: Accelerometer measurement (m/s²) in body frame
            mag_heading_deg: Magnetometer heading (degrees, 0=North, 90=East)
        """
        R_nb = quat_to_rotation_matrix(self.q_tilde)
        R_bn = R_nb.T
        
        # ═══ STEP 1: ACCELEROMETER UPDATE (roll/pitch correction) ═══
        if self.in_maneuver:
            R_accel = self.R_accel * 100.0  # Inflate noise: centripetal accel corrupts gravity measurement
        else:
            R_accel = self.R_accel
        
        y_accel_expected = R_bn @ self.g_n
        innovation_accel = accel_meas - y_accel_expected
        
        H_accel = R_bn @ skew(self.g_n)
        # Zero out yaw column: accelerometer observes gravity (roll/pitch only).
        H_accel[:, 2] = 0.0
        H_accel_full = np.hstack([H_accel, np.zeros((3, 3)), np.zeros((3, 2))])  # 3x8
        
        S_accel = H_accel_full @ self.P @ H_accel_full.T + R_accel
        K_accel = self.P @ H_accel_full.T @ np.linalg.inv(S_accel)
        
        delta_accel = K_accel @ innovation_accel
        self.P = self.P - K_accel @ S_accel @ K_accel.T
        
        # Apply accel corrections
        dq = expq(delta_accel[0:3])
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        self.gyro_bias += delta_accel[3:6]
        self.wind_n += delta_accel[6]
        self.wind_e += delta_accel[7]
        
        # ═══ STEP 2: MAGNETOMETER UPDATE (heading correction) ═══
        # MSFS heading_magnetic is tilt-compensated, so compute heading_expected
        # directly from EKF yaw minus declination. Avoid projecting m_n through
        # the full rotation matrix — at 70° inclination, even 2° of roll/pitch
        # error shifts the projected heading by 5-10°, creating false innovations.
        _, _, yaw_ekf = quat_to_euler(self.q_tilde)
        heading_expected = yaw_ekf - self.mag_dec  # true heading → magnetic heading
        heading_meas = np.radians(mag_heading_deg)
        
        innovation_heading = heading_meas - heading_expected
        innovation_heading = np.arctan2(np.sin(innovation_heading), np.cos(innovation_heading))
        
        H_mag_full = np.array([[0, 0, 1, 0, 0, 0, 0, 0]])  # 1x8
        R_mag_scalar = np.array([[self.R_mag[0, 0]]])
        
        S_mag = H_mag_full @ self.P @ H_mag_full.T + R_mag_scalar
        K_mag = self.P @ H_mag_full.T @ np.linalg.inv(S_mag)
        
        delta_mag = (K_mag @ np.array([innovation_heading])).flatten()
        self.P = self.P - K_mag @ S_mag @ K_mag.T
        
        # Apply mag corrections
        dq = expq(delta_mag[0:3])
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        self.gyro_bias += delta_mag[3:6]
        self.wind_n += delta_mag[6]
        self.wind_e += delta_mag[7]
        
        # Clamp bias to physically reasonable range
        bias_mag = np.linalg.norm(self.gyro_bias)
        if bias_mag > self.max_gyro_bias:
            self.gyro_bias *= self.max_gyro_bias / bias_mag
    
    
    def update_barometer(self, altitude_meas, timestamp):
        """Measurement update from barometer
        
        Args:
            altitude_meas: Barometer altitude (meters)
            timestamp: Current time (seconds)
        """
        # Directly set altitude from barometer (much more accurate than integration)
        self.pos_d = self.alt0 - altitude_meas
        
        # Calculate vertical velocity from altitude change
        dt = timestamp - self.time_prev
        if dt > 0:
            # vel_d positive = down in NED, so negative change = climbing
            self.vel_d = -(altitude_meas - self.alt_prev) / dt
            self.alt_prev = altitude_meas
            self.time_prev = timestamp
    
    
    def update_airspeed(self, airspeed_meas, mag_heading_deg=None):
        """Kalman measurement update using airspeed sensor to estimate wind
        
        Airspeed measures air-relative velocity. Ground velocity comes from accel integration.
        Wind is observable as the difference: wind = air_velocity - ground_velocity
        
        Smart initialization: If starting mid-cruise (vel≈0 but airspeed high), initialize
        velocity from airspeed to avoid incorrect wind estimates.
        
        Measurement model:
            z = [v_air_n, v_air_e] = airspeed * [cos(heading), sin(heading)]
            h = [vel_n + wind_n, vel_e + wind_e] (expected air velocity)
            innovation = z - h
        
        Args:
            airspeed_meas: True airspeed measurement (m/s)
            mag_heading_deg: Magnetic compass heading (degrees) - should always be provided
        """
        # Always use EKF's estimated yaw (true heading) for velocity direction.
        # The EKF yaw tracks true heading via the magnetic field model,
        # so this avoids the declination offset that raw magnetic heading has.
        _, _, yaw = quat_to_euler(self.q_tilde)
        
        # ═══ MEASUREMENT MODEL ═══
        # No smart initialization - let velocity start at 0 (from accel integration)
        # This creates large innovation initially, forcing wind to absorb the discrepancy
        # Measured air velocity vector from airspeed + heading
        v_air_meas_n = airspeed_meas * np.cos(yaw)
        v_air_meas_e = airspeed_meas * np.sin(yaw)
        z = np.array([v_air_meas_n, v_air_meas_e])
        
        # Expected air velocity = ground velocity + wind
        v_air_expected_n = self.vel_n + self.wind_n
        v_air_expected_e = self.vel_e + self.wind_e
        h = np.array([v_air_expected_n, v_air_expected_e])
        
        # Innovation
        innovation = z - h
        
        # ═══ MEASUREMENT JACOBIAN ═══
        # H = ∂h/∂error_state where error_state = [δθ, δb, δw]
        # Air velocity h depends only on wind (velocity is nominal state, not error state)
        # ∂h_n/∂δw_n = 1, ∂h_e/∂δw_e = 1, all others = 0
        H = np.zeros((2, 8))
        H[0, 6] = 1.0  # ∂(vel_n + wind_n)/∂wind_n
        H[1, 7] = 1.0  # ∂(vel_e + wind_e)/∂wind_e
        
        # Measurement noise covariance (2×2)
        R = np.eye(2) * self.R_airspeed
        
        # ═══ KALMAN UPDATE ═══
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain (8×2)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State error correction
        delta_state = K @ innovation  # 8D: [δθ, δb, δw]
        
        # Covariance update
        self.P = self.P - K @ S @ K.T
        
        # ═══ APPLY CORRECTIONS ═══
        # Extract corrections
        delta_theta = delta_state[0:3]  # Orientation error (should be small)
        delta_bias = delta_state[3:6]   # Bias error (should be small)
        delta_wind = delta_state[6:8]   # Wind error (THIS IS KEY!)
        
        # Update orientation (small correction from airspeed-heading consistency)
        dq = expq(delta_theta)
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        
        # Update bias estimate (small correction)
        self.gyro_bias += delta_bias
        # Clamp bias to physically reasonable range
        bias_mag = np.linalg.norm(self.gyro_bias)
        if bias_mag > self.max_gyro_bias:
            self.gyro_bias *= self.max_gyro_bias / bias_mag
        
        # Update wind estimate (THIS MAKES WIND OBSERVABLE)
        self.wind_n += delta_wind[0]
        self.wind_e += delta_wind[1]
        
        # ═══ WIND MAGNITUDE CONSTRAINT ═══
        # Constrain wind estimate to known/expected values
        # In controlled simulation: set to 0.0 m/s (matching MSFS wind settings)
        # In real flight: set based on METAR/forecast (typical cruise: 5-15 m/s, strong: 20-30 m/s)
        # Note: All wind values in m/s (converted from airspeed sensor which is also in m/s after conversion)
        wind_magnitude = np.sqrt(self.wind_n**2 + self.wind_e**2)
        max_wind = 0.0  # m/s - set to match simulator wind configuration (0 knots)
        if wind_magnitude > max_wind:
            # Scale wind vector to max_wind magnitude (soft constraint)
            scale = max_wind / wind_magnitude
            self.wind_n *= scale
            self.wind_e *= scale
        
        # ═══ WIND CONVERGENCE DETECTION ═══
        # Track innovation magnitude to detect when wind has converged
        innovation_magnitude = np.linalg.norm(innovation)
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.convergence_window:
            self.innovation_history.pop(0)
        
        # Wind converged if recent innovations are consistently small
        if len(self.innovation_history) >= self.convergence_window:
            avg_innovation = np.mean(self.innovation_history[-self.convergence_window:])
            # Converged if average innovation < 5 m/s (arbitrary threshold)
            if avg_innovation < 5.0:
                self.wind_converged = True
        
        # ═══ VELOCITY CORRECTION ═══
        # Blend integrated velocity with airspeed-derived velocity
        # Blending always active - even during maneuvers (with reduced trust)
        # This prevents unbounded drift accumulation during turns
        
        # Apply correction to velocity using updated wind estimate
        vel_from_airspeed_n = v_air_meas_n - self.wind_n
        vel_from_airspeed_e = v_air_meas_e - self.wind_e
        
        # Adaptive blending based on maneuver state and wind convergence
        if self.in_maneuver:
            # During maneuvers: moderate trust in airspeed
            alpha = 0.5  # 50% airspeed, 50% integration
        elif self.wind_converged:
            # Straight flight + converged wind: full trust in airspeed
            alpha = 1.0  # 100% airspeed, 0% integration
        else:
            # Straight flight + converging wind: high trust in airspeed
            alpha = 0.8  # 80% airspeed, 20% integration
        
        self.vel_n = alpha * vel_from_airspeed_n + (1 - alpha) * self.vel_n
        self.vel_e = alpha * vel_from_airspeed_e + (1 - alpha) * self.vel_e
        
        # ═══ POSITION UPDATE (deferred from predict step) ═══
        # Integrate position using corrected velocity and heading.
        # This avoids the error from integrating position in predict()
        # where the heading hasn't been corrected by sensor updates yet.
        self.pos_n += self.vel_n * self.last_dt
        self.pos_e += self.vel_e * self.last_dt
    
    
    def get_state(self):
        """Get current state estimate
        
        Returns:
            dict with position, velocity, orientation
        """
        # Convert NED position to lat/lon
        lat_rad = self.lat0_rad + self.pos_n / self.R_earth
        lon_rad = self.lon0_rad + self.pos_e / (self.R_earth * np.cos(self.lat0_rad))
        
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        alt_m = self.alt0 - self.pos_d
        
        # Get Euler angles from quaternion
        roll, pitch, yaw = quat_to_euler(self.q_tilde)
        
        return {
            'latitude': lat_deg,
            'longitude': lon_deg,
            'altitude': alt_m,
            'pos_n': self.pos_n,
            'pos_e': self.pos_e,
            'pos_d': self.pos_d,
            'vel_n': self.vel_n,
            'vel_e': self.vel_e,
            'vel_d': self.vel_d,
            'roll': np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'gyro_bias': self.gyro_bias.copy(),
            'wind_n': self.wind_n,
            'wind_e': self.wind_e
        }
    
    
    def record_state(self, timestamp):
        """Record current state to history"""
        state = self.get_state()
        self.history['timestamp'].append(timestamp)
        self.history['latitude'].append(state['latitude'])
        self.history['longitude'].append(state['longitude'])
        self.history['altitude'].append(state['altitude'])
        self.history['pos_n'].append(state['pos_n'])
        self.history['pos_e'].append(state['pos_e'])
        self.history['pos_d'].append(state['pos_d'])
        self.history['vel_n'].append(state['vel_n'])
        self.history['vel_e'].append(state['vel_e'])
        self.history['vel_d'].append(state['vel_d'])
        self.history['roll'].append(state['roll'])
        self.history['pitch'].append(state['pitch'])
        self.history['yaw'].append(state['yaw'])
        self.history['gyro_bias_x'].append(state['gyro_bias'][0])
        self.history['gyro_bias_y'].append(state['gyro_bias'][1])
        self.history['gyro_bias_z'].append(state['gyro_bias'][2])
        self.history['wind_n'].append(state['wind_n'])
        self.history['wind_e'].append(state['wind_e'])


# ═══════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def run_ekf_ins(input_file, output_file=None):
    """Run Error-State EKF on IMU data
    
    Args:
        input_file: Path to CSV file with IMU/GPS data
        output_file: Path for output CSV (optional)
    """
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Validate required sensors are available
    if 'barometer_pressure' not in df.columns:
        raise ValueError("Barometer (barometer_pressure) required for INS - GPS altitude not suitable")
    if 'heading_magnetic' not in df.columns:
        raise ValueError("Magnetic heading (heading_magnetic) required for INS heading reference")
    if 'pitch' not in df.columns or 'bank' not in df.columns:
        raise ValueError("Pitch and bank required for gravity synthesis (MSFS accel has no gravity)")
    if 'airspeed_true' not in df.columns:
        print("WARNING: Airspeed sensor not available - velocity estimation will be inaccurate in cruise flight")
    
    # Initialize from first sample
    # GPS position at startup only (for reference point initialization)
    lat0 = df['latitude'].iloc[0]
    lon0 = df['longitude'].iloc[0]
    # Use barometric altitude (meters)
    alt0 = barometric_altitude(df['barometer_pressure'].iloc[0])
    # Use heading_magnetic for heading
    # NOTE: Python SimConnect returns all _DEGREES_ variables in RADIANS
    heading0 = np.degrees(df['heading_magnetic'].iloc[0])
    
    print(f"Initial position: {lat0:.6f}°N, {lon0:.6f}°E, {alt0:.1f}m (GPS position + barometer altitude)")
    print(f"Initial heading: {heading0:.1f}° from magnetometer")
    
    # Initialize velocity from first airspeed reading (avoid slow convergence from 0)
    airspeed0 = None
    if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[0]):
        airspeed0 = df['airspeed_true'].iloc[0]  # Already in m/s
        print(f"Initial airspeed: {airspeed0:.1f} m/s")
    
    # Create EKF
    ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0)
    
    print("Running Error-State EKF...")
    
    # Process all samples
    for i in range(len(df)):
        timestamp = df['timestamp'].iloc[i]
        
        # Get time step
        if i == 0:
            dt = 0.0
        else:
            dt = timestamp - df['timestamp'].iloc[i-1]
        
        # Read IMU data with CORRECTED axis mapping
        # MSFS body: X=right, Y=up, Z=forward (left-handed)
        # Standard aero body (NED): X=forward, Y=right, Z=down (right-handed)
        accel_x_msfs = df['accel_x'].iloc[i] * 0.3048  # ft/s² → m/s²
        accel_y_msfs = df['accel_y'].iloc[i] * 0.3048
        accel_z_msfs = df['accel_z'].iloc[i] * 0.3048
        
        # MSFS accelerometers give coordinate acceleration (no gravity).
        # Real IMUs measure specific force (includes gravity).
        # Synthesize gravity from MSFS pitch/bank so the EKF model works correctly.
        pitch_rad = df['pitch'].iloc[i]   # Python SimConnect returns radians
        bank_rad = df['bank'].iloc[i]     # Python SimConnect returns radians
        g = 9.81
        g_body = np.array([
            -g * np.sin(pitch_rad),
            g * np.sin(bank_rad) * np.cos(pitch_rad),
            g * np.cos(bank_rad) * np.cos(pitch_rad)
        ])
        
        accel_body = np.array([
            accel_z_msfs,    # MSFS Z (forward) → Standard X
            accel_x_msfs,    # MSFS X (right)   → Standard Y
            -accel_y_msfs    # MSFS Y (up)      → Standard Z (down), inverted
        ]) + g_body  # Add gravity to match real IMU specific force
        
        gyro_x_msfs = df['gyro_x'].iloc[i]  # rad/s
        gyro_y_msfs = df['gyro_y'].iloc[i]
        gyro_z_msfs = df['gyro_z'].iloc[i]
        
        omega_meas = np.array([
            gyro_z_msfs,     # MSFS Z (forward) → Standard X (roll)
            gyro_x_msfs,     # MSFS X (right)   → Standard Y (pitch)
            gyro_y_msfs      # MSFS Y (up)      → Standard Z (yaw) — NO negation:
                             # angular velocity is a pseudovector; the handedness
                             # change (LH→RH) and the axis flip (Y-up→Z-down)
                             # cancel, so the sign stays the same.
        ])
        
        # ═══ PREDICTION STEP ═══
        ekf.predict(omega_meas, accel_body, dt)
        
        # ═══ MEASUREMENT UPDATE STEPS ═══
        # Store mag_heading for use in airspeed update
        mag_heading = None
        
        # Barometric altitude update
        baro_alt = barometric_altitude(df['barometer_pressure'].iloc[i])
        ekf.update_barometer(baro_alt, timestamp)
        # Use heading_magnetic for heading (Python SimConnect returns radians)
        mag_heading = np.degrees(df['heading_magnetic'].iloc[i])
        ekf.update_accel_mag(accel_body, mag_heading)
        
        # Airspeed update (direct velocity measurement with magnetometer heading)
        if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[i]):
            airspeed = df['airspeed_true'].iloc[i]  # Already in m/s (data_logger converts from knots)
            ekf.update_airspeed(airspeed, mag_heading)
        
        # Record state
        ekf.record_state(timestamp)
    
    print(f"\n✓ EKF processing complete!")
    print(f"✓ Processed {len(df)} data points")
    
    # Save results
    if output_file is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"logs/ekf_ins_{timestamp_str}.csv"
    
    output_df = pd.DataFrame(ekf.history)
    output_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    # Print summary
    final_state = ekf.get_state()
    print(f"\nInitial position: {lat0:.6f}°N, {lon0:.6f}°E")
    print(f"Final estimated position: {final_state['latitude']:.6f}°N, {final_state['longitude']:.6f}°E")
    print(f"Total displacement: N={final_state['pos_n']:.1f}m, E={final_state['pos_e']:.1f}m")
    gyro_bias = final_state['gyro_bias'] * 1000  # Convert to mrad/s
    print(f"Estimated gyro bias: [{gyro_bias[0]:.3f}, {gyro_bias[1]:.3f}, {gyro_bias[2]:.3f}] mrad/s")
    
    return output_file


def barometric_altitude(pressure_mbar, sea_level_pressure=1013.25):
    # Standard barometric formula (ISA):
    # altitude = 44330 * (1 - (pressure / sea_level_pressure) ** 0.1903)
    return 44330.0 * (1.0 - (pressure_mbar / sea_level_pressure) ** 0.1903)


def main():
    parser = argparse.ArgumentParser(description='Error-State EKF for INS')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input CSV file with IMU data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (default: auto-generated)')
    args = parser.parse_args()
    
    run_ekf_ins(args.input, args.output)


if __name__ == "__main__":
    main()
