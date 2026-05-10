# Phase R: Gating Recovery — EKF Relocalization After Innovation Rejection Streak

## Context

Phase A is complete (PF gate, tile plausibility, result writeback, max_visual_innovation_m). GP8 GPS fallback is removed. The pipeline is now GPS-clean.

The live_022_Odense_1ftž run exposed a new failure mode: the EKF innovation gate becomes self-reinforcing once the EKF has drifted. Strong visual measurements are rejected because the EKF sigma underestimates its actual position error (sigma ≈ 43m, actual error ≈ 337m at frame 126). After ~20 frames of continuous rejection, the EKF output is worse than the raw homography alone.

Phase B (homography root-cause fixes: nadir FOV, reproj penalty, cascade order) is **still deferred**. Fix the acceptance/recovery failure first.

---

## Verified Findings

### 1. Frame 126 is the smoking gun (confirmed against trace.json + results.csv)
- CShape=0.81, inliers=348, semantic=0.887, meta_verified=True, verification_matches=467
- EKF pos_sigma=42.88m — but actual EKF position is 337m from corrected homography
- Corrected homography is ~21m from GPS ground truth
- trace.json shows `ekf_before == ekf_after` — no update applied, EKF is frozen wrong

### 2. Long rejection streak F107–F126 (20 consecutive frames) confirmed
- All have `visual_rejected_reason = innovation_too_large`
- All have CShape ≥ 0.79, inliers ≥ 272, meta_verified=1, verification_matches ≥ 328
- EKF sigma grows from 24.95m at F107 → 42.88m at F126 (slow — only process noise)
- EKF actual error grows from ~266m → ~337m (fast — dead reckoning drift at 67 m/s)

### 3. `vel * 1.0` underestimates actual frame interval (confirmed)
- Live run frame interval ≈ 2.4–3.5s, `vel * 1.0` hardcodes 1s allowance
- Frame 126: `max_innovation = max(150, 3×42.88 + 67.2×1.0 + 50) = 246m` vs `vel×dt = 67.2×1.86 = 125m`
- Even with actual dt, frame 126 (337m innovation) still rejected — dt fix helps borderline cases (F102: 169.5m vs 164.6m threshold) but won't break the long streak
- temporal_searcher.py PF gate uses `vel * dt` (actual); run_pipeline.py EKF gate uses `vel * 1.0` — inconsistency confirmed

### 4. `gate_pass` semantics are split across two files (confirmed)
- `temporal_searcher.py:572-574`: returns `gate_pass` = visual quality only (CShape + inliers + homo_position)
- `run_pipeline.py:384-388`: overwrites `result["gate_pass"]` after EKF innovation gate
- trace.json `gate_pass` = final EKF gate decision (written back correctly in Phase A)
- BUT: trace.json does NOT include `visual_innovation_m`, `visual_rejected_reason`, `pf_update_source`, `search_radius_capped` — these are in results.csv only

### 5. Large-R update alone is insufficient to recover (confirmed by math)
- At F126: EKF P ≈ 42.88² ≈ 1839 m², R = 200² = 40000 m² → K ≈ 0.044
- Would move EKF only 0.044 × 337m ≈ 15m — far too small to recover
- Must inflate P first, THEN apply update

### 6. PF is stuck in wrong location (confirmed from trace.json F126)
- n_eff=100.0, spread=33.4m — all 100 particles tightly clustered around wrong EKF position
- PF has been receiving tile_center updates filtered to within max_pf_innovation_m (≈246m) of EKF
- After EKF recovery, PF must be reset — otherwise it continues searching the wrong area

### 7. GP8 removal confirmed — no GPS after init
- run_pipeline.py:405: `ekf.update_position(homo_pos[0], homo_pos[1], R_pos_m2=r_used)` — only called when gate_pass
- No `ekf.update_position(sim_lat, ...)` fallback remains
- GPS columns in results.csv are logging/benchmarking only

### 8. SAVE_PIPELINE_TRACE = True (default) adds ~80-150ms/frame (noted, not changed here)

---

## What the Reviewer Got Wrong or Over-Emphasized

- **"Unify into one shared innovation threshold helper"** — Not necessary. PF gate and EKF gate serve different purposes (PF is looser, EKF is tighter). The actual dt fix is the only consistency issue worth fixing.
- **"Deprecate gate_pass"** — Too much churn. Add `visual_quality_pass` and `ekf_update_applied` alongside existing `gate_pass`.
- **Phase B homography fixes (reprojection penalty, cascade order, nadir FOV)** — Correct diagnosis, wrong priority. Deferred until recovery is working.

---

## Root Cause Diagnosis

The EKF covariance matrix P[8:10] grows only via process noise Q = 25 m²/s per axis. After 20 seconds of no visual correction, sigma grows by only sqrt(25 × 20) ≈ 22m. But velocity at 67 m/s × 20s = 1340m of potential accumulated displacement. The EKF doesn't know it's drifting — it thinks it's 42m uncertain when it's actually 337m off.

The innovation gate formula `max(150, 3σ + v·dt + 50)` is correct in principle, but σ has diverged from reality. The gate needs a recovery mechanism: when strong visual evidence disagrees with the EKF for multiple consecutive frames, stop trusting the EKF sigma and accept the visual measurement.

---

## Implementation Plan

### Stage R1 — Add diagnostics (no behavioral change)

**File: `runtime/run_pipeline.py`**

1. Add to `RESULT_COLUMNS` (after `visual_rejected_reason`):
   ```python
   "visual_quality_pass", "ekf_update_applied",
   "relocalization_candidate", "relocalization_applied",
   ```

2. Add to both `result_row` dicts (file mode line ~447, live mode line ~834):
   ```python
   "visual_quality_pass":       int(bool(result.get("gate_pass_visual_only", False))),
   "ekf_update_applied":        int(bool(r_used is not None)),
   "relocalization_candidate":  int(bool(result.get("relocalization_candidate", False))),
   "relocalization_applied":    int(bool(result.get("relocalization_applied", False))),
   ```
   Note: `gate_pass_visual_only` = `result`'s original `gate_pass` before EKF gate overwrite (snapshot it before line 384).

3. Update `_build_trace_json()` to include:
   ```python
   "visual_innovation_m":       result_row.get("visual_innovation_m"),
   "max_visual_innovation_m":   result_row.get("max_visual_innovation_m"),
   "visual_rejected_reason":    result_row.get("visual_rejected_reason"),
   "pf_update_source":          result_row.get("pf_update_source"),
   "search_radius_capped":      result_row.get("search_radius_capped"),
   "visual_quality_pass":       result_row.get("visual_quality_pass"),
   "ekf_update_applied":        result_row.get("ekf_update_applied"),
   "relocalization_candidate":  result_row.get("relocalization_candidate"),
   "relocalization_applied":    result_row.get("relocalization_applied"),
   ```

---

### Stage R2 — Fix `vel * 1.0` → `vel * dt` in EKF innovation gate

**File: `runtime/run_pipeline.py`**

**File mode (`_process_one_frame`):**
- Add `prev_frame_ts: float = None` parameter
- Compute `dt = max(ts - prev_frame_ts, 0.5) if prev_frame_ts is not None else 1.0`
- Change line ~383: `max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * 1.0 + 50.0)` → `max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * dt + 50.0)`
- In `run_file_mode` loop: track `prev_frame_ts` and pass to `_process_one_frame`

**Live mode (`run_simconnect_mode`, line ~769):**
- Track `prev_frame_ts = None` before loop
- Compute `dt = max(ts - prev_frame_ts, 0.5) if prev_frame_ts is not None else 1.0` each frame
- Update `prev_frame_ts = ts` each frame
- Change line ~769: same substitution as file mode

---

### Stage R3 — Relocalization recovery logic

**File: `config/config.py`** — add section:
```python
# ═══════════════════════════════════════════════════════════════════
# RELOCALIZATION (EKF recovery after innovation rejection streak)
# ═══════════════════════════════════════════════════════════════════

RELOCALIZATION_CONSECUTIVE_THRESHOLD = 5   # strong-rejection frames before recovery attempt
RELOCALIZATION_CSHAPE_MIN = 0.70           # visual quality floor for candidate
RELOCALIZATION_INLIERS_MIN = 200           # inlier count floor for candidate
RELOCALIZATION_VERIFICATION_MIN = 100      # meta-tile verification match floor
RELOCALIZATION_PRIOR_STD_M = 150.0         # inflate P to at least this before recovery update
RELOCALIZATION_R_M = 100.0                 # R for recovery EKF update (std dev in meters)
RELOCALIZATION_COHERENCE_HOP_FACTOR = 5.0  # max consecutive hop = this × vel × dt
```

**File: `runtime/run_pipeline.py`** — both modes

Add state tracking before each mode's main loop:
```python
consecutive_strong_rejections = 0
last_rejected_homo_positions = []   # (lat, lon) ring buffer, max 5 entries
```

In the innovation gate block, after detecting `innovation_too_large`, check if the frame qualifies as a relocalization candidate:
```python
# Is this frame strong enough to be a relocalization candidate?
relocalization_candidate = (
    cs >= config.RELOCALIZATION_CSHAPE_MIN
    and ni >= config.RELOCALIZATION_INLIERS_MIN
    and meta_verified
    and ver_matches >= config.RELOCALIZATION_VERIFICATION_MIN
    and homo_pos is not None
)

if relocalization_candidate:
    consecutive_strong_rejections += 1
    last_rejected_homo_positions.append(homo_pos)
    if len(last_rejected_homo_positions) > 5:
        last_rejected_homo_positions.pop(0)
else:
    consecutive_strong_rejections = 0
    last_rejected_homo_positions.clear()
```

When `gate_pass` is True (accepted normally), reset:
```python
else:
    if gate_pass:
        consecutive_strong_rejections = 0
        last_rejected_homo_positions.clear()
```

After the gate block, check recovery conditions:
```python
relocalization_applied = False
if (not gate_pass
        and relocalization_candidate
        and consecutive_strong_rejections >= config.RELOCALIZATION_CONSECUTIVE_THRESHOLD
        and len(last_rejected_homo_positions) >= 3):
    # Coherence check: consecutive position hops should not exceed velocity-scaled threshold
    recent = last_rejected_homo_positions[-3:]
    avg_dt_for_check = dt if 'dt' in locals() else 1.0
    coherent = True
    for k in range(len(recent) - 1):
        hop = haversine_distance(
            recent[k][0], recent[k][1],
            recent[k+1][0], recent[k+1][1])
        expected_max = config.RELOCALIZATION_COHERENCE_HOP_FACTOR * vel * avg_dt_for_check
        if hop > expected_max:
            coherent = False
            break
    if coherent:
        # Inflate P before update so Kalman gain is meaningful
        ekf.P[8, 8] = max(ekf.P[8, 8],
                          config.RELOCALIZATION_PRIOR_STD_M ** 2)
        ekf.P[9, 9] = max(ekf.P[9, 9],
                          config.RELOCALIZATION_PRIOR_STD_M ** 2)
        ekf.update_position(homo_pos[0], homo_pos[1],
                            R_pos_m2=config.RELOCALIZATION_R_M ** 2)
        gate_pass = True
        visual_rejected_reason = "relocalization_applied"
        relocalization_applied = True
        consecutive_strong_rejections = 0
        last_rejected_homo_positions.clear()
        # Reset PF: force cold start so next frame re-initializes
        # around the recovered EKF position
        searcher.frame_count = 0
```

Write to result dict (after existing writeback block):
```python
result["relocalization_candidate"] = relocalization_candidate
result["relocalization_applied"] = relocalization_applied
```

Note: for **file mode**, this logic needs to move outside `_process_one_frame` since the recovery state is per-run, not per-frame. Options:
- Pass a mutable `recovery_state` dict into `_process_one_frame` and have it mutate it
- OR restructure file mode to match live mode (inline rather than helper function)

**Recommended**: pass a mutable `recovery_state = {}` dict as an extra parameter. `_process_one_frame` receives it, computes `relocalization_candidate`, updates counters inside, and applies recovery update if conditions met. Returns `relocalization_applied` via the `result` dict.

**File: `src/temporal_searcher.py`** — no changes needed
- `searcher.frame_count = 0` is set externally (run_pipeline.py already does this at init time)
- Cold start via `_process_frame_0` will reinitialize PF around the recovered EKF position
- The EKF position (now corrected) is passed in via `imu_data["lat"]` / `imu_data["lon"]`

---

### Expected behavior after fix

At frame 111 (5th consecutive strong rejection after F107):
- Coherence check: F107→F111 positions moving SE at ~67 m/s — consistent hops ≈ 163m each, well within 5 × 67 × 2.5 = 837m threshold ✓
- Covariance inflation: P[8,8] and P[9,9] inflated from 42.88² → 150² (22500 m²)
- Recovery update: K ≈ 22500/(22500 + 10000) ≈ 0.69, moves EKF ~69% of the way from EKF to homo_pos
- `searcher.frame_count = 0` — next frame (F112) is a cold start around corrected EKF position
- F112 cold start resets PF around recovered position

Desired outcome:
- Rejection streak F107–F126 breaks at F111
- Final EKF error much lower than 108.4m
- Corrected homography median (41.0m) becomes achievable target for EKF fusion

---

## Critical Files

| File | Role |
|---|---|
| `runtime/run_pipeline.py:310–455` | `_process_one_frame()` — file mode, innovation gate + EKF update |
| `runtime/run_pipeline.py:633–961` | `run_simconnect_mode()` — live mode, inline innovation gate |
| `runtime/run_pipeline.py:173–231` | `_build_trace_json()` — trace output (missing new fields) |
| `runtime/run_pipeline.py:62–76` | `RESULT_COLUMNS` — add 4 new columns |
| `config/config.py` | Add relocalization constants section |
| `src/temporal_searcher.py:602–614` | `_build_cascade()` — Phase B deferred |

---

## Verification Plan

After implementation:
```powershell
# 1. Unit tests — expect same 26/37 (no regressions)
python -m pytest Pipeline_3_Rev1/tests -q

# 2. Check new columns exist in RESULT_COLUMNS
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'relocalization_applied'

# 3. Live test (user runs)
python Pipeline_3_Rev1/runtime/run_pipeline.py --source simconnect --run-id live_023_recovery_a --max-frames 130
```

After live_023 run, verify:
- `results.csv`: frames 107–126 streak should break (look for `relocalization_applied=1`)
- EKF mean/median error should improve vs live_022 (108.4m mean, 82.2m median)
- `trace.json` for recovery frame should show `visual_rejected_reason=relocalization_applied`
- Simulator GPS still NOT used in `ekf.update_position()` after init (grep for `sim_lat` in any update call)

---

## What Is NOT in Scope

- Phase B (nadir FOV fix, reproj penalty, cascade order, inlier ratio, 3×3 meta-tile)
- Folder restructuring
- Thesis documentation
- Runtime performance (SAVE_PIPELINE_TRACE default = True unchanged for debugging)
- PF measurement noise tuning (MEASUREMENT_NOISE_POSITION_M = 500.0)

---

## Risk Assessment

| Change | Risk | Mitigation |
|---|---|---|
| `vel * dt` fix | Low | Only affects "barely rejected" frames; dt clamped to ≥0.5s |
| New RESULT_COLUMNS | None | Pure logging, no behavioral change |
| trace.json additions | None | Pure logging |
| Recovery covariance inflation | Medium | Thresholds require strong visual evidence; coherence check prevents random wrong homographies |
| `searcher.frame_count = 0` reset | Medium | Cold start is expensive (~2.5s) but safe; only triggers after CONSECUTIVE_THRESHOLD strong rejections |
| Config constants | Low | All tuneable post-run |

The relocalization candidate gate (CShape ≥ 0.70, inliers ≥ 200, meta_verified, vmatches ≥ 100) is conservative. Most bad homographies have either low CShape or low inliers. The main risk is that 5 consecutive BAD homographies that happen to be geometrically coherent trigger a wrong recovery. The coherence hop check mitigates this (bad random matches scatter wildly).
