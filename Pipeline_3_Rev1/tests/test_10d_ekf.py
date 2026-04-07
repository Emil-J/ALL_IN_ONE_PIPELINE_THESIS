"""Quick unit test for 10D EKF expansion."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ekf_ins import ErrorStateEKF, step_ekf, barometric_altitude, preprocess_imu_csv

def test_p_shape():
    ekf = ErrorStateEKF(55.7, 9.5, 100, 180)
    assert ekf.P.shape == (10, 10), f"Expected (10,10), got {ekf.P.shape}"

def test_p_diagonal():
    ekf = ErrorStateEKF(55.7, 9.5, 100, 180)
    assert abs(ekf.P[8, 8] - 200**2) < 1, f"P[8,8]={ekf.P[8,8]}"
    assert abs(ekf.P[9, 9] - 200**2) < 1, f"P[9,9]={ekf.P[9,9]}"

def test_update_position_converges():
    ekf = ErrorStateEKF(55.7, 9.5, 100, 180)
    # Measurement 100m north with R=50²
    target_lat = 55.7 + 100.0 / 6371000 * (180 / np.pi)
    ekf.update_position(target_lat, 9.5, R_pos_m2=50**2)
    s = ekf.get_state()
    shift_m = (s['latitude'] - 55.7) * np.pi / 180 * 6371000
    # Kalman gain ~ 40000/(40000+2500) = 0.94 → expect ~94m shift
    assert 90 < shift_m < 100, f"Shift={shift_m:.1f}m, expected ~94m"
    # P should shrink
    assert ekf.P[8, 8] < 40000, f"P[8,8]={ekf.P[8,8]}, should have shrunk"

def test_update_position_shrinks_P():
    ekf = ErrorStateEKF(55.7, 9.5, 100, 180)
    p_before = ekf.P[8, 8]
    ekf.update_position(55.7, 9.5, R_pos_m2=100**2)
    p_after = ekf.P[8, 8]
    assert p_after < p_before, f"P should shrink: {p_before} -> {p_after}"

def test_predict_grows_P():
    ekf = ErrorStateEKF(55.7, 9.5, 100, 180)
    ekf.update_position(55.7, 9.5, R_pos_m2=50**2)  # shrink first
    p_after_update = ekf.P[8, 8]
    omega = np.array([0.0, 0.0, 0.0])
    accel = np.array([0.0, 0.0, 9.81])
    ekf.predict(omega, accel, dt=1.0)
    p_after_predict = ekf.P[8, 8]
    assert p_after_predict > p_after_update, f"P should grow: {p_after_update} -> {p_after_predict}"

def test_batch_backward_compat():
    """Ensure preprocess_imu_csv still works."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..',
                            'Logs_Run_20260321_162024', 'imu_gps_log_20260321_162024.csv')
    if not os.path.exists(csv_path):
        print("SKIP: CSV not found")
        return
    df = preprocess_imu_csv(csv_path)
    assert len(df) == 970, f"Expected 970 rows, got {len(df)}"
    assert 'latitude_est' in df.columns
    assert 'longitude_est' in df.columns

if __name__ == '__main__':
    tests = [test_p_shape, test_p_diagonal, test_update_position_converges,
             test_update_position_shrinks_P, test_predict_grows_P,
             test_batch_backward_compat]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
    print("Done.")
