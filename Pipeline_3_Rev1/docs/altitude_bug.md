# Altitude Bug — PRESSURE_ALTITUDE Double-Conversion

**Date found**: 2026-04-25  
**Affected runs**: live_008, live_009  
**Symptom**: `altitude_m` ≈ 159 m while `gps_alt_m` ≈ 521 m (ratio ≈ 0.3048)

---

## Root Cause

`step_ekf()` in `src/ekf_ins.py` was changed on 2026-04-23 to prefer
`PRESSURE_ALTITUDE` from SimConnect over `barometer_pressure + ISA formula`.
The change applied `pa * 0.3048` treating `PRESSURE_ALTITUDE` as feet.

However, the Python SimConnect library (`odwdinc/Python-SimConnect`) returns
`PRESSURE_ALTITUDE` already in **metres** (SI units), not feet — despite the
MSFS SDK native unit being feet. Multiplying by 0.3048 applied the conversion
twice, yielding ~30% of the correct altitude (e.g. 521 m → 159 m).

`PLANE_ALTITUDE` (used for `gps_alt_m` via `row["altitude"] * 0.3048`) is
returned in feet by the same library — the two altitude variables are
**inconsistently unitised** within Python SimConnect. This is a library quirk,
not an MSFS SDK issue.

### Calculation trace (live_008, before fix)

```
alt0     = barometric_altitude(barometer_pressure ≈ 952 mbar) ≈ 521 m   ← correct
pa       = PRESSURE_ALTITUDE from SimConnect                            ≈ 521 (metres, not feet)
baro_alt = pa * 0.3048 = 521 * 0.3048                                  ≈ 159 m  ← WRONG
pos_d    = alt0 - baro_alt = 521 - 159                                  = 362 m
altitude = alt0 - pos_d   = 521 - 362                                   = 159 m  ← output
```

---

## Evidence

| Run | `altitude_m` (EKF) | `gps_alt_m` (GPS) | ratio |
|-----|--------------------|-------------------|-------|
| live_007 (old code, barometer_pressure path) | ~540 m | ~540 m | 1.00 ✓ |
| live_008 (new code, pressure_altitude × 0.3048) | ~159 m | ~521 m | 0.305 ✗ |
| live_009 (new code, pressure_altitude × 0.3048) | ~160 m | ~521 m | 0.307 ✗ |

`ratio ≈ 0.3048` for the broken runs — exactly the double-conversion signature.

---

## Fix

### `src/ekf_ins.py` — `step_ekf()` barometer block

**Before (broken):**
```python
baro_alt = None
pa = row.get('pressure_altitude')
if pa is not None and not (isinstance(pa, float) and math.isnan(pa)):
    baro_alt = pa * 0.3048          # ← pa is already metres; * 0.3048 is wrong
else:
    bp = row.get('barometer_pressure')
    if bp is not None and not (isinstance(bp, float) and math.isnan(bp)):
        baro_alt = barometric_altitude(bp)
```

**After (fixed):**
```python
baro_alt = None
bp = row.get('barometer_pressure')
if bp is not None and not (isinstance(bp, float) and math.isnan(bp)):
    baro_alt = barometric_altitude(bp)          # mbar → metres via ISA formula
else:
    pa = row.get('pressure_altitude')
    if pa is not None and not (isinstance(pa, float) and math.isnan(pa)):
        baro_alt = float(pa)                    # already metres (Python SimConnect)
```

`barometer_pressure → barometric_altitude()` is now the primary path (confirmed
working in all live runs before the change). `pressure_altitude` is kept as a
fallback but used directly without any unit conversion.

### `runtime/simconnect_adapter.py` — unit comments

Updated the unit contract docstring and the inline comment on the poll line:
- `pressure_altitude — feet` → `pressure_altitude — metres (Python SimConnect returns SI, NOT feet)`

---

## Impact

Only `altitude_m` in `results.csv` was affected. EKF horizontal position
(`final_lat`, `final_lon`, `pos_n`, `pos_e`) is unaffected because
`update_barometer()` writes only to `pos_d` (vertical) and `vel_d`.

---

## Prevention

**Do not use `pressure_altitude * 0.3048`** in any new code. The variable is
in metres from the Python SimConnect library. If you need feet, divide by
0.3048. `barometer_pressure` (in mbar) → `barometric_altitude()` (in metres)
is the preferred barometric altitude source.
