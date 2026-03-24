# MSFS SimConnect Unit Analysis and Proposed Fixes
**Date:** March 12, 2026 (updated March 21, 2026)  
**Analysis:** Units used in code vs MSFS SimConnect defaults

---

## ✅ Resolution Status (March 21, 2026)

**All critical unit issues have been FIXED:**

1. **AIRSPEED_TRUE** ✅ FIXED: `data_logger.py` now converts knots → m/s before CSV storage. `ekf_ins.py` and `dead_reckoning.py` read airspeed as m/s directly (no double conversion).
2. **GROUND_VELOCITY** ✅ FIXED: Converted to m/s in data_logger.py.
3. **VERTICAL_SPEED** ✅ FIXED: Converted to m/s in data_logger.py.
4. **PLANE_HEADING_DEGREES_MAGNETIC** ⚠️ NEW DISCOVERY: Python SimConnect auto-converts all `_DEGREES_` SimVars to **radians**. The CSV column `heading_magnetic` contains radians, not degrees. Code must call `np.degrees()`.
5. **PLANE_PITCH_DEGREES / PLANE_BANK_DEGREES** ⚠️ Same radian quirk. These are now used for gravity synthesis and stored as radians.
6. **Acceleration axis mapping** ✅ FIXED: No sign inversion in data_logger.py. Axis remapping (MSFS→NED body) done in algorithms with correct signs.
7. **Gyro axis mapping** ✅ FIXED: Angular velocity (pseudovector) requires NO negation on any axis when converting from MSFS LH to Standard RH frame. Previous code incorrectly negated omega_z (yaw).

**Original analysis below remains useful for understanding the issues, but all "CRITICAL ERROR" items have been resolved.**

---

## Executive Summary

**CRITICAL ISSUE FOUND:** Your code assumes `AIRSPEED_TRUE` is in **meters per second (m/s)**, but MSFS SimConnect returns it in **KNOTS** by default. This causes your airspeed measurements to be off by a factor of **1.944 (51% too low)**, which directly impacts wind estimation and velocity convergence.

**Impact:**
- Wind estimates will converge to ~51% of actual wind speed
- Ground velocity estimates will be incorrect
- Position drift will be significantly worse than expected

---

## Detailed Analysis by SimVar

### 1. ACCELERATION_BODY_X/Y/Z ✅ CORRECT

**MSFS Default Unit:** `Feet per second squared` (ft/s²)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 93-95)
accel_x = aq.get("ACCELERATION_BODY_X")  # Returns ft/s²
accel_y = aq.get("ACCELERATION_BODY_Y")
accel_z = aq.get("ACCELERATION_BODY_Z")

# ekf_ins.py (line 711-713)
accel_x_msfs = df['accel_x'].iloc[i] * 0.3048  # ft/s² → m/s²
accel_y_msfs = df['accel_y'].iloc[i] * 0.3048
accel_z_msfs = df['accel_z'].iloc[i] * 0.3048
```

**Status:** ✅ **CORRECT** - You multiply by 0.3048 to convert ft/s² → m/s²

**Conversion Factor Used:** 0.3048  
**Correct Factor:** 0.3048 ✅

---

### 2. ROTATION_VELOCITY_BODY_X/Y/Z ✅ CORRECT

**MSFS Default Unit:** `Radians per second` (rad/s)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 97-99)
gyro_x = aq.get("ROTATION_VELOCITY_BODY_X")  # Returns rad/s
gyro_y = aq.get("ROTATION_VELOCITY_BODY_Y")
gyro_z = aq.get("ROTATION_VELOCITY_BODY_Z")

# ekf_ins.py - No conversion applied
gyro_x_msfs = df['gyro_x'].iloc[i]  # rad/s (no conversion)
```

**Status:** ✅ **CORRECT** - No conversion needed, both code and MSFS use rad/s

**Conversion Factor Used:** None (1.0)  
**Correct Factor:** None (1.0) ✅

---

### 3. PRESSURE_ALTITUDE ✅ CORRECT

**MSFS Default Unit:** `Feet` (ft)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 113)
pressure_altitude = aq.get("PRESSURE_ALTITUDE")  # Returns feet MSL

# ekf_ins.py (line 688)
alt0 = df['pressure_altitude'].iloc[0] * 0.3048  # feet to meters

# ekf_ins.py (line 745)
baro_alt = df['pressure_altitude'].iloc[i] * 0.3048  # feet to meters
```

**Status:** ✅ **CORRECT** - You multiply by 0.3048 to convert feet → meters

**Conversion Factor Used:** 0.3048  
**Correct Factor:** 0.3048 ✅

---

### 4. AIRSPEED_TRUE ❌ **CRITICAL ERROR**

**MSFS Default Unit:** `Knots` (kts)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 122)
airspeed_true = aq.get("AIRSPEED_TRUE")  # Returns KNOTS, NOT m/s!

# ekf_ins.py (line 750)
airspeed = df['airspeed_true'].iloc[i]  # Comment says "Already in m/s in MSFS"
ekf.update_airspeed(airspeed, mag_heading)  # ❌ WRONG UNITS!
```

**Status:** ❌ **INCORRECT** - You assume m/s but MSFS returns KNOTS

**Conversion Factor Used:** None (assumes 1.0)  
**Correct Factor:** 0.514444 (knots → m/s) ❌

**Error Magnitude:**
- 1 knot = 0.514444 m/s
- If airspeed is 120 knots:
  - **Your code uses:** 120 m/s (WRONG)
  - **Should be:** 120 × 0.514444 = 61.73 m/s
  - **Error:** 94.7% too high! (or 48.6% too low when inverted)

---

### 5. AIRSPEED_INDICATED ⚠️ ALSO WRONG (if used)

**MSFS Default Unit:** `Knots` (kts)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 121)
airspeed_indicated = aq.get("AIRSPEED_INDICATED")  # Returns KNOTS

# Currently only logged but not used in calculations
# But if you use it later, needs conversion!
```

**Status:** ⚠️ **Not currently used in EKF, but needs conversion if used**

**Conversion Factor Needed:** 0.514444 (knots → m/s)

---

### 6. GROUND_VELOCITY ⚠️ ALSO WRONG (if used)

**MSFS Default Unit:** `Knots` (kts)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 123)
ground_velocity = aq.get("GROUND_VELOCITY")  # Returns KNOTS

# Currently only logged for display, not used in EKF
```

**Status:** ⚠️ **Used for display only, shows incorrect values but doesn't affect calculations**

**Conversion Factor Needed:** 0.514444 (knots → m/s) for correct display

---

### 7. VERTICAL_SPEED ⚠️ ALSO WRONG (if used)

**MSFS Default Unit:** `Feet per minute` (ft/min)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

**Your Code:**
```python
# data_logger.py (line 124)
vertical_speed = aq.get("VERTICAL_SPEED")  # Returns ft/min

# Currently only logged for display, not used in EKF
```

**Status:** ⚠️ **Used for display only, shows incorrect values but doesn't affect calculations**

**Conversion Factor Needed:** 0.00508 (ft/min → m/s) for correct display  
Or: Divide by 196.85 (ft/min → m/s)

---

### 8. PLANE_LATITUDE/LONGITUDE ✅ CORRECT

**MSFS Default Unit:** `Degrees` (decimal degrees)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Position_State.htm

**Your Code:**
```python
# data_logger.py (line 103-104)
latitude = aq.get("PLANE_LATITUDE")   # Returns degrees
longitude = aq.get("PLANE_LONGITUDE")  # Returns degrees

# Used directly in degrees - no conversion needed
```

**Status:** ✅ **CORRECT** - Both code and MSFS use decimal degrees

---

### 9. PLANE_ALTITUDE ⚠️ FEET (GPS altitude)

**MSFS Default Unit:** `Feet` (ft)  
**Documentation:** https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Position_State.htm

**Your Code:**
```python
# data_logger.py (line 105)
altitude = aq.get("PLANE_ALTITUDE")  # Returns feet

# Used only for ground truth comparison, not in calculations
# But displayed as if it were meters!
```

**Status:** ⚠️ **Used for display/ground truth only, but displayed incorrectly in logs**

**Conversion Factor Needed:** 0.3048 (feet → meters) for correct display

---

## Impact Analysis

### Critical Impact: AIRSPEED_TRUE ❌

This is the **most critical error** affecting your wind estimation algorithm:

**Problem Chain:**
1. MSFS returns airspeed = 120 knots (typical cruise)
2. Your code interprets this as 120 m/s
3. **Actual value:** 120 knots = 61.73 m/s
4. **Your algorithm sees:** 120 m/s (94.7% too high)

**Effect on Wind Estimation:**
```
Example scenario:
- True ground speed: 63 m/s
- True airspeed: 120 knots = 61.73 m/s
- True wind: 61.73 - 63 = -1.27 m/s (nearly calm)

What your code calculates:
- Measured "airspeed": 120 m/s (wrong!)
- Integrated velocity: ~63 m/s (from GPS/accel)
- Estimated wind: 120 - 63 = 57 m/s (COMPLETELY WRONG!)

Your document mentions estimating ~57 m/s wind - this matches exactly!
The wind isn't real - it's a unit conversion artifact!
```

**From your documentation (WIND_ESTIMATION_IMPLEMENTATION.md):**
> Performance Results:
> Wind: ~57 m/s (should be ~57 m/s)

This "57 m/s expected wind" is actually the unit conversion error:
- 120 knots (airspeed) - 63 m/s (ground speed) = **57 m/s apparent "wind"**
- But 120 knots = 61.73 m/s, so actual wind ≈ -1.3 m/s (nearly calm!)

---

## Proposed Solutions

### Solution 1: Fix in data_logger.py (Recommended) ⭐

**Advantages:**
- Fixes at source - logged data is in correct units
- All downstream code works correctly
- Existing CSV files remain incorrect (need reprocessing)

**Implementation:**
```python
# data_logger.py - after line 122
airspeed_indicated = aq.get("AIRSPEED_INDICATED")  # knots
airspeed_true = aq.get("AIRSPEED_TRUE")  # knots
ground_velocity = aq.get("GROUND_VELOCITY")  # knots
vertical_speed = aq.get("VERTICAL_SPEED")  # ft/min

# Convert to SI units immediately
airspeed_indicated_ms = airspeed_indicated * 0.514444  # knots → m/s
airspeed_true_ms = airspeed_true * 0.514444  # knots → m/s
ground_velocity_ms = ground_velocity * 0.514444  # knots → m/s
vertical_speed_ms = vertical_speed * 0.00508  # ft/min → m/s

# Store data point (lines 133-160)
data_log.append({
    # ... existing fields ...
    'airspeed_indicated': airspeed_indicated_ms,  # NOW IN m/s
    'airspeed_true': airspeed_true_ms,  # NOW IN m/s
    'ground_velocity': ground_velocity_ms,  # NOW IN m/s
    'vertical_speed': vertical_speed_ms,  # NOW IN m/s
    # ... rest of fields ...
})

# Update display print (line 175)
print(f"  Speed: IAS {airspeed_indicated:.1f}kts, TAS {airspeed_true:.1f}kts, GS {ground_velocity:.1f}kts")
# Keep displaying in knots for pilot familiarity, but store in m/s
```

**Files to modify:**
- `data_logger.py` (lines ~122-124, add conversions)
- `data_logger.py` (lines ~133-160, update data_log)

**Testing:**
- Collect new flight data
- Verify CSV contains m/s values (airspeed_true should be ~62 m/s at 120 knots cruise)

---

### Solution 2: Fix in ekf_ins.py and dead_reckoning.py

**Advantages:**
- Can reprocess existing CSV files without recollecting data
- Minimal changes

**Disadvantages:**
- Requires fixing multiple files
- Easy to miss one location

**Implementation:**
```python
# ekf_ins.py - line 750
if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[i]):
    airspeed_knots = df['airspeed_true'].iloc[i]  # MSFS returns knots
    airspeed_ms = airspeed_knots * 0.514444  # Convert knots → m/s
    ekf.update_airspeed(airspeed_ms, mag_heading)

# dead_reckoning.py - line 247
if 'airspeed_true' in df.columns and pd.notna(df['airspeed_true'].iloc[i]):
    airspeed_knots = df['airspeed_true'].iloc[i]  # MSFS returns knots
    airspeed_ms = airspeed_knots * 0.514444  # Convert knots → m/s
    # ... use airspeed_ms in calculations
```

**Files to modify:**
- `ekf_ins.py` (line ~750)
- `dead_reckoning.py` (line ~247)
- Any other files that read airspeed_true

---

### Solution 3: Request Units from SimConnect (Most Robust) 🏆

**Advantages:**
- Explicitly specifies desired units
- No ambiguity
- Immune to MSFS version changes
- Best practice

**Disadvantages:**
- Requires understanding SimConnect request syntax
- More code changes

**Implementation:**
```python
# data_logger.py - modify SimConnect requests
# Instead of:
airspeed_true = aq.get("AIRSPEED_TRUE")

# Use explicit unit request:
airspeed_true = aq.get("AIRSPEED_TRUE", "meter per second")
# or
airspeed_true = aq.get("AIRSPEED TRUE", "meters per second")

# Similarly for all variables:
accel_x = aq.get("ACCELERATION_BODY_X", "meter per second squared")
pressure_alt = aq.get("PRESSURE_ALTITUDE", "meters")
vertical_speed = aq.get("VERTICAL_SPEED", "meters per second")
# etc.
```

**Reference:**
- SimConnect SDK documentation on unit requests
- Unit strings: https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Simulation_Variable_Units.htm

---

## Recommended Fix Strategy

### Phase 1: Immediate Fix (Solution 2) - For existing data
1. Add conversion in `ekf_ins.py` line 750: `airspeed_ms = airspeed_knots * 0.514444`
2. Add conversion in `dead_reckoning.py` line 247
3. Reprocess existing CSV files
4. Verify wind estimates drop to realistic values (< 10 m/s typically)

### Phase 2: Long-term Fix (Solution 1 + Solution 3) - For new data
1. Modify `data_logger.py` to convert all units to SI immediately after reading
2. Investigate explicit unit requests with SimConnect (Solution 3)
3. Update comments to clarify units stored in CSV
4. Add unit tests to verify conversions

### Phase 3: Validation
1. Collect test flight in calm conditions (no wind)
2. Verify estimated wind ≈ 0 m/s (should be < 5 m/s)
3. Compare to METAR data for actual wind
4. Check position drift improvements

---

## Quick Reference: Conversion Factors

| Variable | MSFS Unit | Your Code Needs | Conversion Factor | Status |
|----------|-----------|-----------------|-------------------|--------|
| ACCELERATION_BODY_* | ft/s² | m/s² | × 0.3048 | ✅ Correct |
| ROTATION_VELOCITY_* | rad/s | rad/s | × 1.0 | ✅ Correct |
| PRESSURE_ALTITUDE | feet | meters | × 0.3048 | ✅ Correct |
| **AIRSPEED_TRUE** | **knots** | **m/s** | **× 0.514444** | ❌ **MISSING** |
| AIRSPEED_INDICATED | knots | m/s | × 0.514444 | ⚠️ Not used |
| GROUND_VELOCITY | knots | m/s | × 0.514444 | ⚠️ Display only |
| VERTICAL_SPEED | ft/min | m/s | × 0.00508 | ⚠️ Display only |
| PLANE_ALTITUDE | feet | meters | × 0.3048 | ⚠️ Ground truth |

**Critical:** 1 knot = 0.514444 meters per second

---

## Expected Performance Improvement

After fixing the airspeed unit conversion:

**Before (current):**
- Estimated wind: ~57 m/s (WRONG - unit conversion artifact)
- Velocity convergence: Poor (120 m/s "airspeed" vs 63 m/s integrated)
- Position drift: ~15-25 m/s (documented in your implementation)

**After (with fix):**
- Estimated wind: ~1-5 m/s (realistic for calm/light winds)
- Velocity convergence: Excellent (61.73 m/s airspeed ≈ 63 m/s ground in calm wind)
- Position drift: **< 5 m/s expected** (10× improvement!)

**Why such improvement?**
Currently your algorithm sees:
- 120 m/s airspeed (wrong)
- 63 m/s integrated velocity (correct)
- Must explain 57 m/s discrepancy → invents "wind"

After fix:
- 61.73 m/s airspeed (correct)
- 63 m/s integrated velocity (correct)
- Only 1.27 m/s discrepancy → small wind/noise correction

---

## Verification Checklist

After implementing fixes:

### Data Collection
- [ ] New CSV files have airspeed_true ≈ 60-65 m/s (not 120)
- [ ] Display still shows knots in terminal (for pilot familiarity)
- [ ] Other velocities also in m/s (ground_velocity, vertical_speed)

### Algorithm Behavior
- [ ] Wind estimates converge to < 10 m/s
- [ ] Velocity estimates match GPS ground speed
- [ ] Innovation (measured - predicted) is small (< 5 m/s)

### Position Accuracy
- [ ] Position drift rate < 10 m/s after 5 minutes
- [ ] Trajectory matches GPS ground truth within 500m
- [ ] No 17+ km drifts like in current version

### Sanity Checks
- [ ] Airspeed (TAS) > Ground Speed in headwind ✓
- [ ] Airspeed (TAS) < Ground Speed in tailwind ✓
- [ ] Wind vector points in meteorologically reasonable direction
- [ ] Compare to METAR weather data for validation

---

## References

1. **MSFS SimConnect Documentation:**
   - Aircraft SimVars: https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm
   - Simulation Variable Units: https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Simulation_Variable_Units.htm

2. **Unit Conversion Constants:**
   - 1 foot = 0.3048 meters (exact)
   - 1 knot = 0.514444 meters/second (exact: 1852/3600)
   - 1 foot/min = 0.00508 meters/second (exact: 0.3048/60)

3. **Your Documentation:**
   - `docs/WIND_ESTIMATION_IMPLEMENTATION.md` - Section "Performance Expectations"
   - Expected wind estimate: ~57 m/s (this is the unit error!)

---

## Conclusion

**Primary Issue:** AIRSPEED_TRUE unit conversion missing (knots → m/s)

**Impact:** Catastrophic - false wind estimates of ~57 m/s causing poor position accuracy

**Fix Effort:** Low - single line conversions in 1-2 files

**Expected Improvement:** ~10× better position accuracy after convergence

**Recommendation:** Implement Solution 2 immediately for existing data, then Solution 1 for future data collection.

---

**Status:** ✅ **FIXES IMPLEMENTED** (March 12, 2026)

**Changes Applied:**
1. ✅ `ekf_ins.py` line 750-751: Added knots → m/s conversion (0.514444)
2. ✅ `dead_reckoning.py` line 247-248: Added knots → m/s conversion (0.514444)
3. ✅ `data_logger.py` lines 121-130: Convert all speeds at source to SI units
4. ✅ `data_logger.py` display: Keep showing knots/fpm for pilot familiarity
5. ✅ Added documentation header clarifying unit conversions

**Next Steps:**
- Reprocess existing CSV files (will automatically use new conversions)
- Collect new flight data (will be stored in correct units)
- Verify wind estimates drop to realistic values (< 10 m/s)
- Compare position accuracy improvement
