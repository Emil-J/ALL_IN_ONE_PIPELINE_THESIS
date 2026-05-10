# Claude Code Brief — live_023_Odense_1ft Runtime Fix + View-Corrected Visual Update Plan

## Compact prompt to paste into Claude Code

```text
Read the file `docs/LIVE_023_ODENSE_DEBUG_BRIEF.md` first.

I need a small, source-code-grounded fix/plan based on the live_023_Odense_1ft flight test evidence.

Priorities:
1. Fix the runtime crash first: `NameError: name 'ver_matches' is not defined` in `runtime/run_pipeline.py` inside `run_simconnect_mode`, around the relocalization condition.
2. Keep the fix small. Do not refactor the whole pipeline. Do not change EKF math or thresholds unless explicitly required.
3. Add robust extraction of verification quality values (`verification_matches`, `meta_tile_verified`, `tiles_tested`) from the `result` dictionary and/or `meta_tile_info`, so both live and file mode cannot crash from missing local variables.
4. Before telling me the work is finished, always run syntax/import checks and unit tests. At minimum: `python -m py_compile runtime/run_pipeline.py` from inside `Pipeline_3_Rev1`, then `pytest tests -q`. If these commands fail, report the failure and stop.
5. After fixing the crash, inspect semantic alignment. The trace shows the rotated RGB frame goes to SP+LG while the original frame goes to the semantic model. Determine whether semantic confirmation is only histogram-based or spatial. Propose a small plan to rotate the predicted semantic mask with nearest-neighbor interpolation before spatial semantic comparison, while still saving both original and rotated masks for trace/debug.
6. Then inspect turn/bank handling. The goal is not to reject all banked frames. The query frame should still be used. For moderate bank, estimate the camera ground footprint from EKF position + roll/pitch/heading/altitude and use it for tile selection or meta-tile reuse. For hard bank only, limit visual EKF updates.
7. Propose a switch-state design: `NORMAL_TRACKING`, `VIEW_CORRECTED_REUSE`, `VIEW_CORRECTED_RESEARCH`, and `VISUAL_HOLD`. Frame 113 is likely `VIEW_CORRECTED_REUSE`; frame 118 is likely `VIEW_CORRECTED_RESEARCH` or `VISUAL_HOLD`.
8. Do not implement the full view-correction model yet until roll/pitch/heading sign conventions, altitude reference, camera mounting angle, and homography measurement point are verified.

Deliverables:
- First: patch the `ver_matches` crash only.
- Second: report exact files changed.
- Third: show commands run and their outputs.
- Fourth: provide a short implementation plan for semantic-mask rotation and view-corrected tile selection/reuse.
```

---

# Detailed technical brief for Claude Code

## 1. Context

This project is an MSc thesis pipeline for GPS-denied UAV/aircraft localization using:

- MSFS / SimConnect live input
- IMU / barometer / airspeed / heading data
- Semantic segmentation
- SuperPoint + LightGlue feature matching
- Homography-based visual localization
- Error-State EKF sensor fusion
- TMS reference-map tiles and semantic prediction tiles

The latest live run was:

```text
run_id = live_023_Odense_1ft
source = simconnect
frames written = 131
```

The pipeline worked through frame 130, then crashed because of a variable-scope bug in the new relocalization logic.

The current objective is **not a broad refactor**. The immediate goal is:

1. Fix the runtime crash.
2. Verify the code syntactically and with unit tests.
3. Analyze two next design issues:
   - semantic-mask orientation/alignment
   - bank/turn-aware visual update and tile selection

---

## 2. Immediate runtime error to fix

### 2.1 Terminal evidence

The terminal output ends with:

```text
Traceback (most recent call last):
  File "...\Pipeline_3_Rev1\runtime\run_pipeline.py", line 1165, in <module>
    main()
  File "...\Pipeline_3_Rev1\runtime\run_pipeline.py", line 1161, in main
    run_simconnect_mode(args, run_dir, run_id)
  File "...\Pipeline_3_Rev1\runtime\run_pipeline.py", line 890, in run_simconnect_mode
    and ver_matches >= cfg_r.RELOCALIZATION_VERIFICATION_MIN
        ^^^^^^^^^^^
NameError: name 'ver_matches' is not defined
```

### 2.2 Diagnosis

`ver_matches` is used inside the live-mode relocalization condition but is not guaranteed to be defined before use.

This is not an algorithmic localization failure. The run successfully processed 131 frames. The crash is a coding/scope problem in `run_simconnect_mode`.

### 2.3 Required fix style

Fix this with a small robust helper or local extraction block.

Recommended helper concept:

```python
def _extract_meta_quality(result: dict) -> dict:
    meta = result.get("meta_tile_info") or {}

    raw_matches = result.get("verification_matches", None)
    if raw_matches is None:
        raw_matches = meta.get("verification_matches", 0)

    raw_verified = result.get("meta_tile_verified", None)
    if raw_verified is None:
        raw_verified = meta.get("verified", False)

    raw_tiles = result.get("tiles_tested", None)
    if raw_tiles is None:
        raw_tiles = meta.get("tiles_tested", 0)

    try:
        verification_matches = int(raw_matches or 0)
    except (TypeError, ValueError):
        verification_matches = 0

    try:
        tiles_tested = int(raw_tiles or 0)
    except (TypeError, ValueError):
        tiles_tested = 0

    return {
        "verification_matches": verification_matches,
        "meta_tile_verified": bool(raw_verified),
        "tiles_tested": tiles_tested,
    }
```

Then in live mode before the relocalization condition:

```python
mq = _extract_meta_quality(result)
ver_matches = mq["verification_matches"]
meta_verified = mq["meta_tile_verified"]
tiles_tested = mq["tiles_tested"]
```

Use this same extraction style wherever the code writes result rows or checks relocalization conditions.

### 2.4 Strict validation requirement

Before saying the task is finished, run:

```powershell
cd C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Pipeline_3_Rev1
python -m py_compile runtime\run_pipeline.py
pytest tests -q
```

If file mode is available and not too slow, also run a short smoke test:

```powershell
python runtime\run_pipeline.py --source file --run-id smoke_ver_matches_fix --max-frames 30
```

If `--max-frames` is not supported, do **not** add it just for this smoke test. Report that the CLI does not support it.

---

## 3. What the live_023_Odense_1ft run shows

The run summary from `live_analysis.pdf` reports:

```text
Frames: 131
GT frames: 131 / 131
Gate passes: 120 / 131 = 92%
Mean online EKF error: 72.4 m
Median online EKF error: 59.1 m
Minimum: 5.4 m
Maximum: 194.8 m
```

Timing/latency is a major issue:

```text
mean inference latency ≈ 2416 ms
median ≈ 2434 ms
P90 ≈ 2712 ms
max ≈ 3305 ms
mean drone displacement during inference ≈ 158.6 m
```

This means live navigation output is delayed by roughly 2.4 seconds, during which the aircraft can move approximately 150–160 m. This is not the crash, but it is a serious performance/real-time-navigation issue.

---

## 4. Selected representative frames

The pipeline trace selected these frames:

| Frame | Meaning | Error | CShape | Inliers | Semantic conf | Gate |
|---:|---|---:|---:|---:|---:|---|
| 74 | Best visual update | 5.4 m | 0.775 | 618 | 0.780 | PASS |
| 113 | Worst gate-pass visual update | 194.8 m | 0.591 | 618 | 0.768 | PASS |
| 67 | Best no-gate/fallback frame | 13.3 m | 0.228 | 6 | 0.870 | FAIL |
| 118 | Worst no-gate/fallback frame | 152.8 m | 0.579 | 534 | 0.766 | FAIL |

These frames are critical for diagnosis.

---

## 5. Frame 113 is not necessarily a bad visual match

Frame 113 has strong visual evidence:

```json
{
  "frame_idx": 113,
  "gate_pass": true,
  "rotation_deg": 151.59,
  "cs_shape": 0.5908,
  "inliers": 618,
  "verification_matches": 676,
  "semantic_conf": 0.7677,
  "visual_innovation_m": 181.1,
  "max_visual_innovation_m": 238.9,
  "ekf_update_applied": true
}
```

The query and meta-tile can look visually consistent. The problem may not be wrong tile selection. The problem may be that during banked flight, the camera ground footprint is displaced from the aircraft nadir point.

For frame 113, the IMU data gives approximately:

```text
bank = 0.254 rad ≈ 14.6 deg
pressure altitude ≈ 503 m
```

A rough lateral footprint displacement from roll alone is:

```text
offset ≈ altitude * tan(bank)
       ≈ 503 * tan(14.6 deg)
       ≈ 131 m
```

This is the same order as the 150–200 m aircraft-position error. Therefore, the visual match may be correct, but the conversion from visible ground patch to aircraft position may be biased during banking.

### Important distinction

Do not call frame 113 simply a “bad visual match”. More accurate:

> Frame 113 may be a strong image-to-map match but a poor aircraft-position update because the camera footprint is displaced from the aircraft nadir during banked flight.

---

## 6. Frame 118 is harder and may remain rejected

Frame 118 has:

```json
{
  "frame_idx": 118,
  "gate_pass": false,
  "rotation_deg": -173.97,
  "cs_shape": 0.5789,
  "inliers": 534,
  "verification_matches": 598,
  "semantic_conf": 0.7656,
  "visual_innovation_m": 297.5,
  "max_visual_innovation_m": 254.6,
  "visual_rejected_reason": "innovation_too_large",
  "ekf_update_applied": false
}
```

The IMU data gives approximately:

```text
bank = 0.340 rad ≈ 19.5 deg
pressure altitude ≈ 503 m
```

Rough roll-induced footprint shift:

```text
offset ≈ 503 * tan(19.5 deg)
       ≈ 178 m
```

Frame 118 was rejected because the visual innovation exceeded the allowed maximum. This is probably correct. It should be evaluated after view correction, but it may still be too risky and may belong to `VIEW_CORRECTED_RESEARCH` or `VISUAL_HOLD`.

---

## 7. Semantic-mask orientation/alignment issue

The trace explicitly states:

```text
Rotated frame -> SP+LG matcher | Original frame -> semantic model
```

This means:

- SuperPoint + LightGlue receives the heading-rotated RGB frame.
- The semantic model receives the original, unrotated frame.

This is not automatically wrong if semantic confirmation is only a global histogram, because class histograms are mostly rotation-insensitive.

However, if semantic confirmation is supposed to help spatial matching, then the semantic mask must be aligned with the same orientation as the reference/meta-tile.

### Recommended design

Do **not** necessarily run the UNet on the rotated RGB image first. That may introduce black-corner distribution shift.

Better:

```text
raw query RGB
  -> semantic model
  -> predicted semantic class mask
  -> rotate predicted semantic mask using same heading transform
  -> use nearest-neighbor interpolation
  -> compare rotated semantic mask against reference/meta-tile semantic mask
```

Implementation note:

```python
cv2.warpAffine(..., flags=cv2.INTER_NEAREST)
```

Never use bilinear interpolation for semantic class labels.

### Trace/debug output should save both

```text
semantic_mask_original.png
semantic_mask_rotated.png
```

Original = model output quality.
Rotated = spatial semantic confirmation.

---

## 8. Bank/turn logic: corrected requirement

The goal is **not** to reject all visual updates during turns.

Correct goal:

> During moderate bank, still use the query frame, but predict/correct the camera ground footprint before choosing or reusing tiles and before converting homography output into aircraft position.

### Current simplified logic

```text
EKF aircraft position
  -> choose nearby TMS tiles
  -> match query to tiles/meta-tile
  -> homography ground position
  -> fixed look-ahead correction
  -> EKF update if gate passes
```

### Better view-corrected logic

```text
EKF aircraft position + roll + pitch + heading + altitude
  -> predicted camera ground footprint
  -> choose/reuse tiles around predicted footprint
  -> match query to tiles/meta-tile
  -> homography estimates observed ground footprint
  -> convert observed footprint back to aircraft nadir/position
  -> innovation gate
  -> EKF update if valid
```

---

## 9. Proposed switch-state design

This is a design plan. Do **not** implement fully before sign conventions are verified.

| State | Meaning | Tile handling | EKF update |
|---|---|---|---|
| `NORMAL_TRACKING` | Straight/mild attitude | Current PF/meta-tile logic | Normal visual update |
| `VIEW_CORRECTED_REUSE` | Moderate bank + strong current match | Reuse existing top tiles/meta-tile | Apply view/nadir correction, then innovation gate |
| `VIEW_CORRECTED_RESEARCH` | Moderate bank + weak/failed reuse | Search around predicted camera footprint | Apply correction, stricter gate |
| `VISUAL_HOLD` | Hard bank/extreme geometry | Optional matching for diagnostics | No visual EKF update |

### Expected frame classification

- Frame 113: likely `VIEW_CORRECTED_REUSE` because visual match is strong but aircraft-position estimate is biased.
- Frame 118: likely `VIEW_CORRECTED_RESEARCH` or `VISUAL_HOLD` because visual innovation is too large.

---

## 10. View/nadir correction concept

### 10.1 Estimate camera-footprint offset

Use EKF state and attitude:

```text
aircraft_lat, aircraft_lon
heading/yaw
roll/bank
pitch
altitude
```

Simple first-order model:

```text
side_offset    ≈ altitude * tan(roll_effective)
forward_offset ≈ altitude * tan(pitch_or_camera_tilt_effective)
```

Rotate into North/East:

```text
dN = forward_offset * cos(heading) - side_offset * sin(heading)
dE = forward_offset * sin(heading) + side_offset * cos(heading)
```

### 10.2 Before tile selection

For moderate bank:

```text
predicted_footprint = aircraft_position + [dN, dE]
```

Use this predicted footprint as the search centre instead of raw EKF aircraft position.

### 10.3 After homography

If homography gives the observed ground footprint:

```text
aircraft_visual_estimate = homography_ground_position - [dN, dE]
```

Then use the existing innovation gate.

### 10.4 Critical warning

Do not implement this blindly. First verify:

- roll sign convention
- pitch sign convention
- heading convention
- altitude reference: pressure altitude vs altitude AGL vs MSL
- camera mounting angle
- whether the homography point represents image centre, projected nadir, inlier centroid, weighted centroid, or projected point
- where current fixed look-ahead correction is applied

A wrong sign can double the error.

---

## 11. Reuse-before-research policy

For a frame like 113, do not immediately do a broad new search.

Suggested order:

```text
1. Use/reuse current top tiles/meta-tile.
2. If match is strong but innovation/position error is suspicious, apply view/nadir correction.
3. If corrected innovation passes, use visual update.
4. If corrected innovation still fails, search around predicted footprint.
5. If still bad or hard-bank state, hold visual EKF update and coast with EKF.
```

This is more efficient and less fragile than broad re-searching every banked frame.

---

## 12. Do not over-trust semantic confidence yet

Semantic confidence was high for both good and bad cases:

```text
Frame 74:  sem_conf ≈ 0.780, error ≈ 5.4 m
Frame 113: sem_conf ≈ 0.768, error ≈ 194.8 m
Frame 67:  sem_conf ≈ 0.870, gate fail, error ≈ 13.3 m
Frame 118: sem_conf ≈ 0.766, gate fail, error ≈ 152.8 m
```

Therefore, current `semantic_conf` is not sufficient as a reliability detector. It is probably closer to class-distribution similarity than a true spatial consistency metric.

If semantic should help reject bad spatial matches, use a spatially aligned semantic mask, not only a global histogram.

---

## 13. Existing innovation gate already does part of the job

The system already rejects visual updates that are too far from the EKF. Frame 118 is an example:

```text
visual_innovation_m = 297.5 m
max_visual_innovation_m = 254.6 m
visual_rejected_reason = innovation_too_large
ekf_update_applied = false
```

So do not duplicate this logic blindly.

The new view-corrected logic should improve the visual aircraft-position estimate **before** the innovation gate.

---

## 14. Hard-bank limit still needed

Even with footprint correction, there should be a hard-bank limit.

Suggested initial policy:

```text
|roll| < 15 deg:
    allow bank-aware visual update

15 deg <= |roll| < 25 deg:
    allow only if visual quality is strong, meta-tile verified, and corrected innovation passes; inflate R

|roll| >= 25 deg:
    VISUAL_HOLD: no EKF visual position update
```

These thresholds are first guesses. Do not hard-code them without checking the current config style and results.

---

## 15. What to inspect in the code

Inspect exact insertion points in:

```text
Pipeline_3_Rev1/runtime/run_pipeline.py
Pipeline_3_Rev1/src/temporal_searcher.py
Pipeline_3_Rev1/src/particle_filter.py
Pipeline_3_Rev1/src/meta_tile_builder.py
Pipeline_3_Rev1/src/best_first_search.py
Pipeline_3_Rev1/src/visual_measurement.py
Pipeline_3_Rev1/src/position_estimator.py
Pipeline_3_Rev1/src/semantic_confirmer.py
Pipeline_3_Rev1/src/semantic_tile_scorer.py
Pipeline_3_Rev1/config/config.py
```

Specific questions:

1. Where exactly is tile search centre chosen?
2. Does it currently use raw EKF aircraft lat/lon?
3. Where does `homo_lat/homo_lon` come from?
4. Which homography measurement method is selected?
5. What point is converted to lat/lon: image centre, inlier centroid, weighted centroid, projected point, or another method?
6. Where is fixed look-ahead correction applied?
7. Is roll already used anywhere besides scaling the look-ahead correction or R?
8. Does semantic confirmation use global histograms, spatial comparison, or both?
9. Where can `semantic_mask_rotated.png` be saved in the pipeline trace?
10. How can `visual_update_mode` and `visual_rejected_reason` be written to `results.csv`/trace without breaking analysis notebooks?

---

## 16. Expected deliverables for this task

### Deliverable A — crash fix

- Fix `ver_matches` undefined crash.
- Keep the patch minimal.
- Report exact files changed.
- Explain why the crash happened.

### Deliverable B — validation

Before saying done, run:

```powershell
cd C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Pipeline_3_Rev1
python -m py_compile runtime\run_pipeline.py
pytest tests -q
```

Report outputs.

If smoke test is possible:

```powershell
python runtime\run_pipeline.py --source file --run-id smoke_ver_matches_fix --max-frames 30
```

If `--max-frames` is unsupported, report it. Do not add CLI flags unless asked.

### Deliverable C — semantic alignment plan

- Confirm whether semantic is histogram-only or spatial.
- Propose where to rotate predicted semantic masks.
- Specify nearest-neighbor interpolation.
- Specify trace files to save.

### Deliverable D — view-corrected tracking plan

- Confirm current search centre.
- Propose switch-state design.
- Identify exact insertion points.
- Evaluate likely behavior on frame 113 and frame 118.
- Do not implement full correction before sign conventions are verified.

---

## 17. Non-goals for this task

Do not do these unless explicitly approved:

- Broad refactor
- Moving files/folders
- Renaming modules
- Changing EKF math
- Changing thresholds broadly
- Rewriting semantic model logic
- Rewriting temporal searcher architecture
- Implementing full camera geometry correction before sign conventions are verified
- Removing the innovation gate

---

## 18. Final desired behavior

After this task, the pipeline should at minimum:

1. No longer crash because of undefined `ver_matches`.
2. Pass syntax checks.
3. Pass unit tests.
4. Have a clear follow-up plan for semantic-mask rotation.
5. Have a clear follow-up plan for bank-aware view-corrected tile reuse/search.
6. Preserve the current working live pipeline behavior except for the crash fix.
