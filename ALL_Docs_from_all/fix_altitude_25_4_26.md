
Pipeline 3 — Altitude Bug Fix: PRESSURE_ALTITUDE Double-Conversion
Context
User reported constant / wrong altitude in live_008 and live_009 runs. altitude_m column shows ~159 m while gps_alt_m shows ~521 m. The ratio 521/159 ≈ 3.28 = 1/0.3048 is the feet-to-metres conversion factor — a classic sign of the same unit conversion being applied twice.

The bug was introduced in the previous session (2026-04-23) as an "optional cleanup" that switched step_ekf() to prefer PRESSURE_ALTITUDE × 0.3048 over barometric_altitude(barometer_pressure). But the Python SimConnect library returns PRESSURE_ALTITUDE already in metres (not feet), so multiplying by 0.3048 shrinks the value by a factor of 3.28.

Root-Cause Analysis
Evidence from live data
Run	altitude_m (EKF)	gps_alt_m (GPS)	ratio
live_007 (pre-fix, old code)	~540 m	~540 m	1.00 ✓
live_008 (post-fix, new code)	~159 m	~521 m	0.305 ✗
live_009 (post-fix, new code)	~160 m	~521 m	0.307 ✗
0.305 ≈ 0.3048 — altitude_m = gps_alt_m × 0.3048. Exactly the symptom of treating a metres value as feet and multiplying by 0.3048.

Code path (ekf_ins.py:step_ekf, lines 545-554)
# BUG — introduced 2026-04-23
baro_alt = None
pa = row.get('pressure_altitude')
if pa is not None and not (isinstance(pa, float) and math.isnan(pa)):
    baro_alt = pa * 0.3048          # ← pa is already metres; * 0.3048 is wrong
else:
    bp = row.get('barometer_pressure')
    if bp is not None and not (isinstance(bp, float) and math.isnan(bp)):
        baro_alt = barometric_altitude(bp)   # ← this path is correct
if baro_alt is not None:
    ekf.update_barometer(baro_alt, timestamp)
update_barometer() sets pos_d = alt0 - altitude_meas, so altitude_m = alt0 - pos_d = altitude_meas = baro_alt. When baro_alt = 521 * 0.3048 = 159 m, the output is 159 m instead of 521 m.

Why the old path (barometer_pressure) was correct
barometric_altitude(pressure_mbar) uses the ISA formula and returns metres from the raw pressure in mbar. Confirmed working in live_007 (altitude_m ≈ 540 m ≈ gps_alt_m).

Why PRESSURE_ALTITUDE appears in metres
The Python SimConnect library (odwdinc/Python-SimConnect) returns PRESSURE_ALTITUDE in metres (SI units), not feet, despite the MSFS SDK native unit being feet. PLANE_ALTITUDE (which populates gps_alt_m via row["altitude"] * 0.3048) is returned in feet, making the two variables inconsistently unitised within the same library.

The comment in simconnect_adapter.py — # feet (PRESSURE_ALTITUDE) — is therefore wrong and must be corrected.

Impact
Only altitude_m in results.csv is affected. The EKF position states (pos_n, pos_e, final_lat, final_lon) are unaffected because update_barometer() only writes to pos_d and vel_d.

Fix
File 1 — src/ekf_ins.py (line 545–554)
Swap preference: use barometer_pressure → barometric_altitude() first (the known-working path), keep pressure_altitude as fallback but use it directly without × 0.3048 (it is already metres).

# FIXED
baro_alt = None
bp = row.get('barometer_pressure')
if bp is not None and not (isinstance(bp, float) and math.isnan(bp)):
    baro_alt = barometric_altitude(bp)          # mbar → metres via ISA formula
else:
    pa = row.get('pressure_altitude')
    if pa is not None and not (isinstance(pa, float) and math.isnan(pa)):
        baro_alt = float(pa)                    # already metres (SimConnect Python)
if baro_alt is not None:
    ekf.update_barometer(baro_alt, timestamp)
No other changes to ekf_ins.py.

File 2 — runtime/simconnect_adapter.py (line 294 comment)
Correct the unit annotation from # feet to # metres (SimConnect Python returns SI):

# Before:
slow_cache["pressure_altitude"] = aq.get("PRESSURE_ALTITUDE")   # feet

# After:
slow_cache["pressure_altitude"] = aq.get("PRESSURE_ALTITUDE")   # metres (Python SimConnect returns SI)
Also correct the docstring at the top of the file (lines 30–31):

# Before:
pressure_altitude  — feet    (PRESSURE_ALTITUDE)

# After:
pressure_altitude  — metres  (PRESSURE_ALTITUDE; Python SimConnect returns SI, not feet)
File 3 — docs/altitude_bug.md (NEW)
Create a bug-analysis document recording the root cause and fix for future reference. See content below.

New file: Pipeline_3_Rev1/docs/altitude_bug.md
# Altitude Bug — PRESSURE_ALTITUDE Double-Conversion

**Date found**: 2026-04-25  
**Affected runs**: live_008, live_009  
**Symptom**: altitude_m ≈ 159 m while gps_alt_m ≈ 521 m (ratio ≈ 0.3048)

## Root Cause

`step_ekf()` in `src/ekf_ins.py` was changed (2026-04-23) to prefer
`PRESSURE_ALTITUDE` from SimConnect over `barometer_pressure + ISA formula`.
The code applied `pa * 0.3048` treating `PRESSURE_ALTITUDE` as feet.

However, the Python SimConnect library (`odwdinc/Python-SimConnect`) returns
`PRESSURE_ALTITUDE` already in **metres**, not feet. The × 0.3048 factor was
applied twice, yielding ~30% of the correct altitude.

`PLANE_ALTITUDE` (used for `gps_alt_m`) is returned in feet by the same
library — the two variables have inconsistent units within Python SimConnect.

## Evidence

altitude_m / gps_alt_m ≈ 0.3048 for both live_008 and live_009.
live_007 (using the old barometer_pressure path) showed altitude_m ≈ gps_alt_m ✓.

## Fix (commit: ...)

1. `src/ekf_ins.py:step_ekf()` — reverted barometer block to use
   `barometer_pressure → barometric_altitude()` as primary path.
   Fallback to `pressure_altitude` kept but WITHOUT × 0.3048.

2. `runtime/simconnect_adapter.py` — corrected unit comment for
   `pressure_altitude` from `feet` to `metres (SimConnect Python returns SI)`.

## Impact

Only `altitude_m` in `results.csv` was affected. EKF lat/lon output is
unaffected (update_barometer writes only to pos_d/vel_d, not pos_n/pos_e).
Files Modified Summary
File	Change
src/ekf_ins.py	Revert barometer block in step_ekf() — swap preference, remove × 0.3048 from pressure_altitude fallback
runtime/simconnect_adapter.py	Correct unit comments for pressure_altitude (feet → metres)
docs/altitude_bug.md	NEW — bug analysis and fix documentation
Verification
Offline unit check (no MSFS needed)
# In Pipeline_3_Rev1/ with the project venv
python -c "
from src.ekf_ins import ErrorStateEKF, step_ekf
ekf = ErrorStateEKF(55.7, 9.5, 521.0, 90.0)
row = {
    'timestamp': 1.0, 'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
    'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0, 'pitch': 0, 'bank': 0,
    'heading_magnetic': 1.57,
    'barometer_pressure': 952.0,   # ~521 m altitude
}
step_ekf(ekf, row, None)
state = ekf.get_state()
print(f'altitude={state[\"altitude\"]:.1f} m  (expect ~521)')
# Should print ~521, not ~159
"
Cross-check with live data
After a new SimConnect run, verify altitude_m in results.csv is within ~5 m of gps_alt_m. Previously live_007 showed <1 m residual using the barometer_pressure path.

Regression check
File-mode runs are unaffected (they use barometer_pressure in _run_ekf_core and _init_ekf). Confirm altitude_m values are consistent with prior file-mode results.