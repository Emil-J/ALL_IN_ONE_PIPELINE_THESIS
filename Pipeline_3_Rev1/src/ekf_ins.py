"""
Error-State Extended Kalman Filter for INS
Based on Algorithm 4 from Kok, Hol, Schön (2017)
"Using Inertial Sensors for Position and Orientation Estimation"

EKF class copied from MSFS2020_IMU_Pipeline/ekf_ins.py — math kept identical.

Entry point:
  preprocess_imu_csv(csv)     — Pipeline 3 wrapper, returns DataFrame with
                                 unambiguous estimate vs ground-truth columns
"""

import numpy as np
import pandas as pd
import warnings


# ═══════════════════════════════════════════════════════════════════
# QUATERNION UTILITIES
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

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

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


def barometric_altitude(pressure_mbar, sea_level_pressure=1013.25):
    """Standard barometric formula (ISA)"""
    return 44330.0 * (1.0 - (pressure_mbar / sea_level_pressure) ** 0.1903)


# ═══════════════════════════════════════════════════════════════════
# ERROR-STATE EKF (Algorithm 4 from paper)
# ═══════════════════════════════════════════════════════════════════

class ErrorStateEKF:
    """
    Error-State Extended Kalman Filter for orientation estimation

    State: 3D orientation error (delta_theta)
    Linearization point: Quaternion q_tilde (orientation estimate)

    Measurements:
    - Gyroscope: angular velocity
    - Accelerometer: gravity direction (for pitch/roll)
    - Magnetometer: magnetic north (for yaw)
    - Barometer: altitude
    """

    def __init__(self, lat0, lon0, alt0, heading0, airspeed0=None):
        self.R_earth = 6371000.0
        self.lat0_rad = np.radians(lat0)
        self.lon0_rad = np.radians(lon0)

        self.q_tilde = quat_from_euler(0.0, 0.0, np.radians(heading0))
        self.mag_dec = np.radians(4.0)  # ~4 deg East at Vejle, Denmark

        self.pos_n = 0.0
        self.pos_e = 0.0
        self.pos_d = 0.0
        self.alt0 = alt0

        if airspeed0 is not None and airspeed0 > 1.0:
            yaw_rad = np.radians(heading0) + self.mag_dec
            self.vel_n = airspeed0 * np.cos(yaw_rad)
            self.vel_e = airspeed0 * np.sin(yaw_rad)
        else:
            self.vel_n = 0.0
            self.vel_e = 0.0
        self.vel_d = 0.0

        self.gyro_bias = np.zeros(3)
        self.max_gyro_bias = 0.015

        self.wind_n = 0.0
        self.wind_e = 0.0

        self.alt_prev = alt0
        self.time_prev = 0.0

        self.maneuver_threshold = 2.0
        self.in_maneuver = False
        self.last_dt = 0.0

        self.wind_converged = False
        self.innovation_history = []
        self.convergence_window = 50

        # Covariance: 10D [orientation error (3), bias error (3), wind (2), position error NE (2)]
        self.P = np.eye(10) * 0.5
        self.P[3:6, 3:6] = np.eye(3) * 0.01
        self.P[6:8, 6:8] = np.eye(2) * (5.0)**2
        self.P[8:10, 8:10] = np.eye(2) * (200.0)**2  # initial position uncertainty

        self.g_n = np.array([0, 0, 9.81])

        mag_inc = np.radians(70.0)
        self.m_n = np.array([
            np.cos(mag_inc) * np.cos(self.mag_dec),
            np.cos(mag_inc) * np.sin(self.mag_dec),
            np.sin(mag_inc)
        ])

        # Process noise
        self.Q_gyro = np.eye(3) * (0.05)**2
        self.Q_gyro_bias = np.eye(3) * (0.03)**2
        self.Q_wind = np.eye(2) * (0.1)**2

        # Measurement noise
        self.R_accel = np.eye(3) * (1.2)**2
        self.R_mag = np.eye(3) * (0.05)**2
        self.R_baro = 1.0**2
        self.R_airspeed = 2.0**2

        # Complementary filter gain for heading
        self.heading_gain_initial = 0.95
        self.heading_gain_steady = 0.85
        self.heading_convergence_samples = 40
        self.heading_correction_gain = self.heading_gain_initial
        self.sample_count = 0

        self.history = {
            'timestamp': [], 'latitude': [], 'longitude': [], 'altitude': [],
            'pos_n': [], 'pos_e': [], 'pos_d': [],
            'vel_n': [], 'vel_e': [], 'vel_d': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'gyro_bias_x': [], 'gyro_bias_y': [], 'gyro_bias_z': [],
            'wind_n': [], 'wind_e': []
        }

    def predict(self, omega_meas, accel_body, dt):
        if dt <= 0:
            return

        omega_corrected = omega_meas - self.gyro_bias
        dq = expq(dt / 2 * omega_corrected)
        self.q_tilde = quat_multiply(self.q_tilde, dq)
        self.q_tilde = quat_normalize(self.q_tilde)

        R_nb = quat_to_rotation_matrix(self.q_tilde)

        G_theta = dt * R_nb

        F = np.eye(10)
        F[0:3, 3:6] = 0.5 * dt * R_nb
        # F[8:10, 8:10] = I_2 (already from np.eye)

        self.P = F @ self.P @ F.T
        self.P[0:3, 0:3] += G_theta @ self.Q_gyro @ G_theta.T
        self.P[3:6, 3:6] += self.Q_gyro_bias * dt
        self.P[6:8, 6:8] += self.Q_wind * dt
        self.P[8:10, 8:10] += np.eye(2) * (5.0)**2 * dt  # position process noise

        # Maneuver detection
        accel_ned = R_nb @ accel_body
        accel_horizontal = np.sqrt(accel_ned[0]**2 + accel_ned[1]**2)
        self.in_maneuver = accel_horizontal > self.maneuver_threshold

        # Velocity update
        accel_true = accel_ned - self.g_n
        self.vel_n += accel_true[0] * dt
        self.vel_e += accel_true[1] * dt

        # Position update deferred to update_airspeed
        self.last_dt = dt

    def update_accel_mag(self, accel_meas, mag_heading_deg):
        R_nb = quat_to_rotation_matrix(self.q_tilde)
        R_bn = R_nb.T

        # Accelerometer update (roll/pitch)
        if self.in_maneuver:
            R_accel = self.R_accel * 100.0
        else:
            R_accel = self.R_accel

        y_accel_expected = R_bn @ self.g_n
        innovation_accel = accel_meas - y_accel_expected

        H_accel = R_bn @ skew(self.g_n)
        H_accel[:, 2] = 0.0
        H_accel_full = np.hstack([H_accel, np.zeros((3, 3)), np.zeros((3, 2)), np.zeros((3, 2))])

        S_accel = H_accel_full @ self.P @ H_accel_full.T + R_accel
        K_accel = self.P @ H_accel_full.T @ np.linalg.inv(S_accel)

        delta_accel = K_accel @ innovation_accel
        self.P = self.P - K_accel @ S_accel @ K_accel.T

        dq = expq(delta_accel[0:3])
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        self.gyro_bias += delta_accel[3:6]
        self.wind_n += delta_accel[6]
        self.wind_e += delta_accel[7]
        self.pos_n += delta_accel[8]
        self.pos_e += delta_accel[9]

        # Magnetometer update (heading)
        _, _, yaw_ekf = quat_to_euler(self.q_tilde)
        heading_expected = yaw_ekf - self.mag_dec
        heading_meas = np.radians(mag_heading_deg)

        innovation_heading = heading_meas - heading_expected
        innovation_heading = np.arctan2(np.sin(innovation_heading), np.cos(innovation_heading))

        H_mag_full = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        R_mag_scalar = np.array([[self.R_mag[0, 0]]])

        S_mag = H_mag_full @ self.P @ H_mag_full.T + R_mag_scalar
        K_mag = self.P @ H_mag_full.T @ np.linalg.inv(S_mag)

        delta_mag = (K_mag @ np.array([innovation_heading])).flatten()
        self.P = self.P - K_mag @ S_mag @ K_mag.T

        dq = expq(delta_mag[0:3])
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        self.gyro_bias += delta_mag[3:6]
        self.wind_n += delta_mag[6]
        self.wind_e += delta_mag[7]
        self.pos_n += delta_mag[8]
        self.pos_e += delta_mag[9]

        bias_mag = np.linalg.norm(self.gyro_bias)
        if bias_mag > self.max_gyro_bias:
            self.gyro_bias *= self.max_gyro_bias / bias_mag

    def update_barometer(self, altitude_meas, timestamp):
        self.pos_d = self.alt0 - altitude_meas
        dt = timestamp - self.time_prev
        if dt > 0:
            self.vel_d = -(altitude_meas - self.alt_prev) / dt
            self.alt_prev = altitude_meas
            self.time_prev = timestamp

    def update_airspeed(self, airspeed_meas, mag_heading_deg=None):
        _, _, yaw = quat_to_euler(self.q_tilde)

        v_air_meas_n = airspeed_meas * np.cos(yaw)
        v_air_meas_e = airspeed_meas * np.sin(yaw)
        z = np.array([v_air_meas_n, v_air_meas_e])

        v_air_expected_n = self.vel_n + self.wind_n
        v_air_expected_e = self.vel_e + self.wind_e
        h = np.array([v_air_expected_n, v_air_expected_e])

        innovation = z - h

        H = np.zeros((2, 10))
        H[0, 6] = 1.0
        H[1, 7] = 1.0

        R = np.eye(2) * self.R_airspeed
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_state = K @ innovation
        self.P = self.P - K @ S @ K.T

        delta_theta = delta_state[0:3]
        delta_bias = delta_state[3:6]
        delta_wind = delta_state[6:8]

        dq = expq(delta_theta)
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)

        self.gyro_bias += delta_bias
        bias_mag = np.linalg.norm(self.gyro_bias)
        if bias_mag > self.max_gyro_bias:
            self.gyro_bias *= self.max_gyro_bias / bias_mag

        self.wind_n += delta_wind[0]
        self.wind_e += delta_wind[1]

        self.pos_n += delta_state[8]
        self.pos_e += delta_state[9]

        # Wind magnitude constraint (0 wind in sim)
        wind_magnitude = np.sqrt(self.wind_n**2 + self.wind_e**2)
        max_wind = 0.0
        if wind_magnitude > max_wind:
            scale = max_wind / wind_magnitude
            self.wind_n *= scale
            self.wind_e *= scale

        # Wind convergence detection
        innovation_magnitude = np.linalg.norm(innovation)
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.convergence_window:
            self.innovation_history.pop(0)
        if len(self.innovation_history) >= self.convergence_window:
            avg_innovation = np.mean(self.innovation_history[-self.convergence_window:])
            if avg_innovation < 5.0:
                self.wind_converged = True

        # Velocity correction
        vel_from_airspeed_n = v_air_meas_n - self.wind_n
        vel_from_airspeed_e = v_air_meas_e - self.wind_e

        if self.in_maneuver:
            alpha = 0.5
        elif self.wind_converged:
            alpha = 1.0
        else:
            alpha = 0.8

        self.vel_n = alpha * vel_from_airspeed_n + (1 - alpha) * self.vel_n
        self.vel_e = alpha * vel_from_airspeed_e + (1 - alpha) * self.vel_e

        # Position update (deferred from predict)
        self.pos_n += self.vel_n * self.last_dt
        self.pos_e += self.vel_e * self.last_dt

    def update_position(self, lat_meas, lon_meas, R_pos_m2=None):
        """Visual position measurement update.

        Converts measured lat/lon → NED, computes innovation against
        current pos_n/pos_e, and runs a standard Kalman update on error
        states [8:10].

        Args:
            lat_meas: Measured latitude (degrees)
            lon_meas: Measured longitude (degrees)
            R_pos_m2: Measurement noise variance in m² (scalar or 2x2).
                       Default 100² = 10 000 m².
        """
        if R_pos_m2 is None:
            R_pos_m2 = 100.0**2

        # Convert measurement to NED relative to EKF origin
        meas_n = (np.radians(lat_meas) - self.lat0_rad) * self.R_earth
        meas_e = (np.radians(lon_meas) - self.lon0_rad) * self.R_earth * np.cos(self.lat0_rad)

        z = np.array([meas_n, meas_e])
        h = np.array([self.pos_n, self.pos_e])
        innovation = z - h

        # H: position error states are at indices 8,9
        H = np.zeros((2, 10))
        H[0, 8] = 1.0
        H[1, 9] = 1.0

        if np.isscalar(R_pos_m2):
            R = np.eye(2) * R_pos_m2
        else:
            R = np.array(R_pos_m2).reshape(2, 2)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_state = K @ innovation
        self.P = self.P - K @ S @ K.T

        # Apply corrections
        dq = expq(delta_state[0:3])
        self.q_tilde = quat_multiply(dq, self.q_tilde)
        self.q_tilde = quat_normalize(self.q_tilde)
        self.gyro_bias += delta_state[3:6]
        self.wind_n += delta_state[6]
        self.wind_e += delta_state[7]
        self.pos_n += delta_state[8]
        self.pos_e += delta_state[9]

    def get_state(self):
        lat_rad = self.lat0_rad + self.pos_n / self.R_earth
        lon_rad = self.lon0_rad + self.pos_e / (self.R_earth * np.cos(self.lat0_rad))

        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        alt_m = self.alt0 - self.pos_d

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
# SHARED EKF PROCESSING CORE
# ═══════════════════════════════════════════════════════════════════

def step_ekf(ekf, row, prev_timestamp=None):
    """Process a single IMU row through the EKF (predict + all sensor updates).

    This is the loop body extracted from _run_ekf_core, suitable for online
    (step-by-step) processing.  Does NOT call record_state() — caller decides.

    Args:
        ekf: ErrorStateEKF instance (mutated in place)
        row: dict-like with keys: timestamp, accel_x/y/z, gyro_x/y/z,
             heading_magnetic, pitch, bank, barometer_pressure,
             optionally airspeed_true
        prev_timestamp: timestamp of the previous row (for dt).
                        If None, dt=0 (first row).

    Returns:
        state dict from ekf.get_state()
    """
    g = 9.81
    timestamp = row['timestamp']
    dt = 0.0 if prev_timestamp is None else (timestamp - prev_timestamp)

    # ── MSFS axis mapping (left-handed body → right-handed NED body) ──
    accel_x_msfs = row['accel_x'] * 0.3048
    accel_y_msfs = row['accel_y'] * 0.3048
    accel_z_msfs = row['accel_z'] * 0.3048

    pitch_rad = row['pitch']
    bank_rad  = row['bank']

    g_body = np.array([
        -g * np.sin(pitch_rad),
         g * np.sin(bank_rad) * np.cos(pitch_rad),
         g * np.cos(bank_rad) * np.cos(pitch_rad)
    ])

    accel_body = np.array([accel_z_msfs, accel_x_msfs, -accel_y_msfs]) + g_body

    omega_meas = np.array([row['gyro_z'], row['gyro_x'], row['gyro_y']])

    # ── Predict ──
    ekf.predict(omega_meas, accel_body, dt)

    # ── Measurement updates ──
    baro_raw = row['barometer_pressure']
    if baro_raw is not None:
        baro_alt = barometric_altitude(baro_raw)
        ekf.update_barometer(baro_alt, timestamp)

    heading_mag_raw = row.get('heading_magnetic')
    if heading_mag_raw is not None:
        mag_heading_deg = np.degrees(heading_mag_raw)
        ekf.update_accel_mag(accel_body, mag_heading_deg)
    else:
        mag_heading_deg = ekf.get_state()['yaw']

    airspeed = row.get('airspeed_true', None)
    if airspeed is not None and not (isinstance(airspeed, float) and np.isnan(airspeed)):
        ekf.update_airspeed(airspeed, mag_heading_deg)

    return ekf.get_state()


def _run_ekf_core(df):
    """
    Run Error-State EKF on a DataFrame of MSFS SimConnect sensor data.

    GPS lat/lon from row 0 ONLY — sets the local NED reference origin.
    All subsequent positions are dead-reckoned from IMU sensors.
    Raw GPS is NEVER fed into later state propagation.

    CSV column units (from Python SimConnect / data_logger):
      accel_x, accel_y, accel_z  — ft/s² (coordinate accel, no gravity)
      gyro_x, gyro_y, gyro_z    — rad/s
      heading_magnetic           — radians (SimConnect _DEGREES_ → radians)
      pitch, bank                — radians (SimConnect _DEGREES_ → radians)
      airspeed_true              — m/s (logger converts from knots)
      barometer_pressure         — mbar
      latitude, longitude        — degrees (GROUND TRUTH — row 0 only)

    Returns:
        ekf: ErrorStateEKF instance with populated history
    """
    # Validate required columns
    required = ['timestamp', 'latitude', 'longitude', 'barometer_pressure',
                'heading_magnetic', 'pitch', 'bank',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"IMU CSV missing required columns: {missing}")

    # ── Initialise from row 0 (GPS for reference origin ONLY) ──
    lat0 = df['latitude'].iloc[0]
    lon0 = df['longitude'].iloc[0]
    alt0 = barometric_altitude(df['barometer_pressure'].iloc[0])
    # NOTE: Python SimConnect returns all _DEGREES_ variables in RADIANS
    heading0 = np.degrees(df['heading_magnetic'].iloc[0])

    airspeed0 = None
    if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[0]):
        airspeed0 = df['airspeed_true'].iloc[0]  # already m/s

    ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0)

    g = 9.81  # m/s²

    for i in range(len(df)):
        timestamp = df['timestamp'].iloc[i]

        if i == 0:
            dt = 0.0
        else:
            dt = timestamp - df['timestamp'].iloc[i - 1]

        # ── MSFS axis mapping (left-handed body → right-handed NED body) ──
        # Accelerometers: ft/s² → m/s²
        accel_x_msfs = df['accel_x'].iloc[i] * 0.3048
        accel_y_msfs = df['accel_y'].iloc[i] * 0.3048
        accel_z_msfs = df['accel_z'].iloc[i] * 0.3048

        pitch_rad = df['pitch'].iloc[i]   # radians from SimConnect
        bank_rad  = df['bank'].iloc[i]    # radians from SimConnect

        # MSFS accel has NO gravity component → synthesise it from attitude
        g_body = np.array([
            -g * np.sin(pitch_rad),
             g * np.sin(bank_rad) * np.cos(pitch_rad),
             g * np.cos(bank_rad) * np.cos(pitch_rad)
        ])

        # MSFS body (X=right, Y=up, Z=forward) → Standard NED body (X=fwd, Y=right, Z=down)
        accel_body = np.array([
            accel_z_msfs,     # MSFS Z (forward) → Standard X
            accel_x_msfs,     # MSFS X (right)   → Standard Y
            -accel_y_msfs     # MSFS Y (up)      → Standard Z (down)
        ]) + g_body

        # Gyroscopes: rad/s, pseudovector sign cancellation (no negation needed)
        omega_meas = np.array([
            df['gyro_z'].iloc[i],  # MSFS Z (forward) → Standard X (roll)
            df['gyro_x'].iloc[i],  # MSFS X (right)   → Standard Y (pitch)
            df['gyro_y'].iloc[i]   # MSFS Y (up)      → Standard Z (yaw)
        ])

        # ── Predict ──
        ekf.predict(omega_meas, accel_body, dt)

        # ── Measurement updates ──
        baro_alt = barometric_altitude(df['barometer_pressure'].iloc[i])
        ekf.update_barometer(baro_alt, timestamp)

        # heading_magnetic is radians from SimConnect → convert to degrees for EKF API
        mag_heading_deg = np.degrees(df['heading_magnetic'].iloc[i])
        ekf.update_accel_mag(accel_body, mag_heading_deg)

        # Airspeed update (already m/s from data_logger)
        if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[i]):
            ekf.update_airspeed(df['airspeed_true'].iloc[i], mag_heading_deg)

        ekf.record_state(timestamp)

    return ekf


# ═══════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════

# Map from internal EKF history keys → unambiguous output column names
_EKF_COL_MAP = {
    'timestamp':   'timestamp',
    'latitude':    'latitude_est',
    'longitude':   'longitude_est',
    'altitude':    'altitude_est',
    'yaw':         'yaw_deg',
    'roll':        'roll_deg',
    'pitch':       'pitch_deg',
    'pos_n':       'pos_n',
    'pos_e':       'pos_e',
    'pos_d':       'pos_d',
    'vel_n':       'vel_n',
    'vel_e':       'vel_e',
    'vel_d':       'vel_d',
    'gyro_bias_x': 'gyro_bias_x',
    'gyro_bias_y': 'gyro_bias_y',
    'gyro_bias_z': 'gyro_bias_z',
    'wind_n':      'wind_n',
    'wind_e':      'wind_e',
}


def _build_estimate_df(ekf):
    """Build estimate-only DataFrame from EKF history with unambiguous names."""
    return pd.DataFrame(ekf.history).rename(columns=_EKF_COL_MAP)


def _check_gps_leakage(est_df, raw_df, n_check=20):
    """Warn loudly if estimated lat/lon are bit-for-bit identical to raw GPS.

    Row 0 is excluded (initialisation makes them identical by design).
    """
    n = min(n_check, len(est_df), len(raw_df))
    if n < 3:
        return
    est_lat = est_df['latitude_est'].values[1:n]
    raw_lat = raw_df['latitude'].values[1:n]
    est_lon = est_df['longitude_est'].values[1:n]
    raw_lon = raw_df['longitude'].values[1:n]
    if np.array_equal(est_lat, raw_lat) and np.array_equal(est_lon, raw_lon):
        warnings.warn(
            "GPS LEAKAGE DETECTED: estimated lat/lon are bit-for-bit identical "
            f"to raw GPS for samples 1..{n-1}. The EKF output is probably just "
            "ground-truth GPS, not dead-reckoned estimates. Check that raw GPS "
            "is not being fed into state propagation.",
            RuntimeWarning,
            stacklevel=3,
        )


# ═══════════════════════════════════════════════════════════════════
# PIPELINE 3 CONVENIENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════

def preprocess_imu_csv(csv_path):
    """Run EKF on IMU CSV and return a DataFrame with UNAMBIGUOUS columns.

    GPS is used ONLY at row 0 to set the NED reference origin.

    Estimate columns (from EKF dead reckoning):
        latitude_est   — estimated latitude  (degrees)
        longitude_est  — estimated longitude (degrees)
        altitude_est   — estimated altitude  (metres)
        yaw_deg        — estimated heading   (degrees, true north)
        vel_n, vel_e   — NED velocity        (m/s)
        pos_n, pos_e, pos_d — NED displacement from origin (metres)

    Ground-truth columns (simulator GPS, clearly labelled):
        gps_lat — raw GPS latitude  from CSV (degrees)
        gps_lon — raw GPS longitude from CSV (degrees)
        gps_alt — raw GPS altitude  from CSV (metres)

    Original sensor columns preserved (timestamp, gyro_*, accel_*, etc.)
    except latitude/longitude/altitude which are renamed to gps_*.
    """
    df = pd.read_csv(csv_path)
    ekf = _run_ekf_core(df)
    est_df = _build_estimate_df(ekf)

    # Start from original CSV, renaming raw GPS so it can't be mistaken
    result = df.rename(columns={
        'latitude':  'gps_lat',
        'longitude': 'gps_lon',
        'altitude':  'gps_alt',
    })

    # Merge estimate columns (skip timestamp — already present)
    for col in est_df.columns:
        if col == 'timestamp':
            continue
        result[col] = est_df[col].values

    _check_gps_leakage(est_df, df)

    return result

