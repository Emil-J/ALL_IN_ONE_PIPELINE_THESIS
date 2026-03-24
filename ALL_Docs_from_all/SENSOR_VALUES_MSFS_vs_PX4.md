
---

## How to Obtain All Sensor Values from PX4 (Real Hardware)

To access the full set of sensor values (including raw magnetometer axes, IMU, barometer, GPS, airspeed, etc.) from PX4 on a real drone, the recommended approach is:

1. **Companion Computer Integration:**
  - Use a companion computer (e.g., Raspberry Pi, Jetson, NUC) connected to the PX4 flight controller via serial, USB, or Ethernet.

2. **uORB Message Access:**
  - PX4 internally uses uORB for all sensor and state messages.
  - To expose desired uORB messages externally, modify the PX4 configuration to publish them over ROS2 using the `uxrdds` bridge.

3. **ROS2 + uXRCDDS Environment:**
  - Set up ROS2 on the companion computer.
  - Use the `uxrdds` bridge to relay uORB messages from PX4 to ROS2 topics.

4. **.yaml Configuration:**
  - Edit the relevant `.yaml` file (e.g., `dds_topics.yaml`) to specify which uORB messages should be published and their topic mapping. (https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/uxrce_dds_client/dds_topics.yaml)(https://docs.px4.io/main/en/msg_docs/SensorBaro)
  - Example: Add `sensor_mag`, `sensor_accel`, `sensor_gyro`, `airspeed`, `barometer`, `vehicle_gps_position`, etc.

5. **PX4 Firmware Reflash:**
  - After modifying the `.yaml` configuration, reflash the PX4 firmware to apply changes and unlock the desired uORB messages for external access.

6. **Data Logging and Processing:**
  - Use ROS2 nodes on the companion computer to subscribe to the published topics and log/process sensor data as needed.

---

**Summary:**

To obtain all sensor values from PX4 for advanced sensor fusion, use a companion computer running ROS2 with the uXRCDDS bridge, modify the `.yaml` configuration to expose required uORB messages, and reflash PX4 firmware. This enables full access to IMU, magnetometer axes, barometer, GPS, airspeed, and more for real-world applications.
# Sensor Values Used in MSFS Simulation vs PX4

## Overview
This document explains which sensor values are used from Microsoft Flight Simulator (MSFS) via SimConnect for the INS/Dead Reckoning pipeline, and why raw magnetometer axes (X, Y, Z) are not available. It also compares the available sensor data to what is typically provided by the PX4 flight controller.

---

## Sensor Values Used from MSFS (SimConnect)

- **IMU (Accelerometer & Gyroscope):**
  - ACCELERATION_BODY_X, ACCELERATION_BODY_Y, ACCELERATION_BODY_Z (ft/s², converted to m/s² in algorithms)
  - ROTATION_VELOCITY_BODY_X, ROTATION_VELOCITY_BODY_Y, ROTATION_VELOCITY_BODY_Z (rad/s)
  - **IMPORTANT:** MSFS accelerometers report coordinate acceleration (NO gravity component). Gravity is synthesized from pitch/bank in the algorithms.

- **GPS:**
  - PLANE_LATITUDE, PLANE_LONGITUDE, PLANE_ALTITUDE (for ground truth only)

- **Barometer:**
  - BAROMETER_PRESSURE (millibars, converted to altitude using ISA formula)

- **Magnetometer/Heading:**
  - PLANE_HEADING_DEGREES_MAGNETIC (preferred — more stable heading from flight model)
  - MAGNETIC_COMPASS (fallback — compass instrument reading)
  - **⚠️ Python SimConnect Radian Quirk:** All SimVars with `_DEGREES_` in the name are auto-converted to **radians** by the Python SimConnect library. The CSV column `heading_magnetic` contains radians, not degrees. Code must call `np.degrees()` before use.

- **Attitude (for gravity synthesis):**
  - PLANE_PITCH_DEGREES, PLANE_BANK_DEGREES (returned in radians by Python SimConnect)
  - These are NOT used as attitude estimates — only for synthesizing the missing gravity component in MSFS accelerometer data.

- **Airspeed:**
  - AIRSPEED_INDICATED, AIRSPEED_TRUE (raw unit: knots; converted to m/s in data_logger.py before CSV storage)

---

## Why Magnetometer X, Y, Z Are Not Used

- **SimConnect Limitation:**
  - MSFS SimConnect does NOT provide raw magnetometer axes (X, Y, Z). Only heading and compass values are available.
  - Typical variables: PLANE_HEADING_DEGREES_MAGNETIC, MAGNETIC_COMPASS.

- **PX4 Comparison:**
  - PX4 flight controller provides raw magnetometer axes (MAG_X, MAG_Y, MAG_Z) via MAVLink.
  - These are used for more accurate heading estimation and sensor fusion in real-world drones.

- **Impact:**
  - In MSFS, heading estimation relies on MAGNETIC_COMPASS or PLANE_HEADING_DEGREES_MAGNETIC.
  - Cannot compute heading from raw axes; must use provided heading values.

---

## If We Could Use Magnetometer X, Y, Z
If MSFS SimConnect provided raw magnetometer axes (X, Y, Z), the implementation would be:

- **Data Logging:**
  - Log MAG_X, MAG_Y, MAG_Z values for each sample.

- **Heading Calculation:**
  - Compute heading as:
    - `heading = np.degrees(np.arctan2(mag_y, mag_x))`
  - This allows direct calculation of yaw from the magnetic field vector, as done in PX4-based sensor fusion.

- **EKF Integration:**
  - Use raw axes in the EKF measurement update for more accurate orientation estimation.
  - Magnetometer axes would be passed to the EKF, enabling full 3D magnetic vector fusion (not just heading).

- **Advantages:**
  - More robust to magnetic disturbances and sensor errors.
  - Matches real-world drone sensor fusion pipelines (PX4, ArduPilot).

- **Limitation:**
  - MSFS SimConnect does not expose these variables, so this approach is not possible in simulation.

---

## Summary Table
| Sensor Type      | MSFS SimConnect Variable(s)         | PX4 MAVLink Variable(s)         | Used in Pipeline? |
|------------------|-------------------------------------|---------------------------------|-------------------|
| Accelerometer    | ACCELERATION_BODY_X/Y/Z             | ACCEL_X/Y/Z                     | Yes (+ gravity synthesis from pitch/bank) |
| Gyroscope        | ROTATION_VELOCITY_BODY_X/Y/Z        | GYRO_X/Y/Z                      | Yes               |
| Barometer        | BAROMETER_PRESSURE                   | BARO_PRESSURE, BARO_ALTITUDE    | Yes               |
| Magnetometer     | HEADING_DEGREES_MAGNETIC, COMPASS   | MAG_X/Y/Z                       | Heading only (no raw axes) |
| Attitude         | PITCH_DEGREES, BANK_DEGREES          | (computed internally)           | Yes (gravity synthesis) |
| GPS              | PLANE_LATITUDE/LONGITUDE/ALTITUDE   | GPS_LAT/LON/ALT                 | Yes (ground truth)|
| Airspeed         | AIRSPEED_INDICATED/TRUE             | AIRSPEED                       | Yes (knots→m/s)   |

---

## Conclusion
- The pipeline uses all available sensor values from MSFS SimConnect.
- Magnetometer X, Y, Z axes are not available in MSFS, so heading estimation uses compass/heading values.
- PX4 provides raw axes, enabling more advanced sensor fusion in real-world applications.
- This limitation is inherent to the MSFS simulation API and cannot be bypassed.
