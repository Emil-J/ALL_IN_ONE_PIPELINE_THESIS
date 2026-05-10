# Code Review: GPS-Clean Pipeline, Innovation-Gate Failure, and Recovery Plan

**Project:** `Pipeline_3_Rev1` — GPS-denied UAV localisation pipeline  
**Prepared for:** Claude Code / repository repair planning  
**Scope:** Code-level diagnosis based on the current modified files and latest run output.  
**Main files reviewed:**

- `runtime/run_pipeline.py`
- `src/temporal_searcher.py`
- `src/visual_measurement.py`
- `src/meta_tile_builder.py`
- `src/particle_filter.py`
- `config/config.py`
- `results.csv` from `live_022_Odense_1ftž`
- trace files for frames `0000`, `0059`, `0082`, `0126`

---

## 0. Executive Summary

The pipeline is now **scientifically cleaner** because the previous simulator-GPS fallback was removed. That is the correct direction for a GPS-denied thesis.

However, the new run exposes a deeper control/fusion problem:

> The EKF innovation gate now rejects many visually strong homography measurements because the EKF estimate has already drifted too far away. The filter then refuses the exact visual measurements that could recover it.

So the current system is no longer "GPS-cheating," but it is now **over-gated and recovery-poor**.

The main failure pattern is visible in `live_022_Odense_1ftž`:

- 127 frames total.
- 86 accepted EKF visual updates.
- 41 rejected frames.
- 36 frames rejected specifically due to `innovation_too_large`.
- 19 rejected frames were visually strong: high `CShape`, many inliers, high semantic confidence, and meta-tile verified.
- From frame 107 to frame 126 there is a long rejection streak.
- Frame 126 is the clearest example: visually strong homography, but EKF refuses to update.

The corrected homography is **better than the EKF overall** in this run:

| Method | Mean error | Median error | Max error |
|---|---:|---:|---:|
| Online EKF | 108.4 m | 82.2 m | 343.6 m |
| Raw homography | 152.5 m | 116.4 m | 961.0 m |
| Corrected homography | 89.4 m | 41.0 m | 851.6 m |

This means the fusion layer is often degrading the visual measurement instead of improving it.

**Blunt conclusion:** the visual localisation module is not the main trash part anymore. The current trash part is the estimator acceptance/recovery logic.

---

## 1. What Is Good in the Current Code

### 1.1 GP8 simulator-GPS fallback appears removed from EKF correction

The new `run_pipeline.py` no longer shows the old logic:

```python
if not gate_pass:
    ekf.update_position(sim_lat, sim_lon, R_pos_m2=200.0**2)
```

That is good. Simulator/GPS lat/lon are still written to `results.csv` as `gps_lat`, `gps_lon`, and `gps_alt_m`, but they are now logging/benchmarking fields, not EKF correction fields.

**Why this matters:** the thesis can now defend the claim "GPS-denied after initial geodetic prior" more honestly, provided future evaluation keeps this removed.

### 1.2 Diagnostic columns are much better

`run_pipeline.py` now includes useful diagnostic columns in `RESULT_COLUMNS` around lines 62–82:

```python
"visual_innovation_m", "max_visual_innovation_m", "visual_rejected_reason",
"pf_update_source", "search_radius_m", "search_radius_capped",
```

This is good because it tells us *why* frames failed. Without this, the run would only say `gate=fail`, which is too vague.

### 1.3 File and live modes now share similar innovation-gate logic

The file-mode helper `_process_one_frame()` computes:

```python
visual_innovation_m = haversine_distance(homo_pos[0], homo_pos[1], ekf_lat, ekf_lon)
max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * 1.0 + 50.0)
if gate_pass and visual_innovation_m > max_innovation_m:
    gate_pass = False
    visual_rejected_reason = "innovation_too_large"
```

Live mode contains the same pattern later in `run_simconnect_mode()`.

This is structurally better than having hidden GPS fallback differences between modes.

### 1.4 Visual localisation can produce strong measurements

Frame 59 is a clean success:

```text
frame_idx = 59
CShape = 0.7698
inliers = 614
semantic_conf = 0.7806
meta_tile_verified = true
r_used_sqrt = 27.32 m
final error ≈ 7.9 m
```

This proves the SP+LG + meta-tile + homography + look-ahead chain can work.

---

## 2. The Main Failure: EKF Innovation Gate Blocks Recovery

### 2.1 The current logic assumes the EKF is more trustworthy than the visual measurement

In both file mode and live mode, `run_pipeline.py` does this:

```python
if gate_pass and visual_innovation_m > max_innovation_m:
    gate_pass = False
    visual_rejected_reason = "innovation_too_large"
```

This logic is safe when the visual measurement is suspicious.

But it becomes wrong when the EKF itself has drifted. Then a large innovation does not mean "visual is bad." It can mean:

> The EKF is wrong and the visual measurement is trying to recover the state.

That is exactly what happens late in the Odense run.

### 2.2 Frame 126 is the smoking gun

Frame 126 has:

```text
CShape = 0.8100
inliers = 348
semantic_conf = 0.8873
meta_tile_verified = true
verification_matches = 467
homo_corrected ≈ (55.356036, 10.413870)
GPS GT ≈ (55.355874, 10.414074)
EKF output ≈ (55.354982, 10.408870)
visual_innovation_m = 337.1 m
max_visual_innovation_m = 246.0 m
r_used_sqrt = null
visual_rejected_reason = innovation_too_large
```

The visual homography is close to ground truth, but the EKF rejects it because the EKF has already drifted too far.

**Simple explanation:** the filter is saying:

> "This visual measurement is too far from me, therefore the visual measurement must be wrong."

But the data says:

> "No, the EKF is the thing that is wrong."

### 2.3 The failure becomes self-reinforcing

Once the EKF starts rejecting visual measurements, it receives no position correction. Then it drifts more. Then the next visual innovation becomes even larger. Then it rejects again.

This creates a bad feedback loop:

```text
EKF drifts
→ visual measurement far away
→ innovation gate rejects visual measurement
→ EKF receives no correction
→ EKF drifts more
→ visual measurement even farther away
→ repeated rejection
```

This is exactly visible in the final run segment:

```text
Frame 102 → 105: rejected due to innovation_too_large
Frame 107 → 126: continuous rejected sequence
```

### 2.4 Strong rejected frames prove the gate is too strict

There are 19 rejected frames that are visually strong:

Criteria used:

```text
gate_pass = 0
CShape > 0.7
inliers > 100
semantic_conf > 0.7
meta_tile_verified = 1
```

Examples:

| Frame | CShape | Inliers | Semantic | Innovation | Max allowed | Homography vs GT behaviour |
|---:|---:|---:|---:|---:|---:|---|
| 102 | 0.7776 | 574 | 0.8083 | 169.5 m | 164.6 m | visually close, barely rejected |
| 103 | 0.7738 | 590 | 0.7686 | 210.0 m | 171.0 m | visually close |
| 118 | 0.8114 | 668 | 0.7824 | 298.0 m | 228.1 m | visually close |
| 123 | 0.8178 | 554 | 0.8197 | 321.0 m | 240.3 m | visually close |
| 126 | 0.8100 | 348 | 0.8873 | 337.1 m | 246.0 m | visually close |

This is not random failure. It is a systematic recovery problem.

---

## 3. The Innovation Threshold Formula Is Too Naive

### 3.1 Current EKF innovation threshold

In `run_pipeline.py`, both file and live mode use:

```python
max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * 1.0 + 50.0)
```

Problems:

1. `vel * 1.0` hard-codes a one-second allowance.
2. The actual visual frame spacing is closer to 2–3 seconds in the live run.
3. The pipeline latency is about 3 seconds.
4. `pos_sigma_now` grows too slowly during long rejection streaks.
5. The gate has no recovery mode.

### 3.2 TemporalSearcher uses a different innovation allowance

In `temporal_searcher.py`, the PF innovation gate uses:

```python
max_pf_innovation_m = max(
    150.0,
    3.0 * imu_data.get("pos_sigma", 0.0)
    + imu_data.get("velocity_mps", 0.0) * dt
    + 50.0
)
```

This is better because it uses the actual inter-frame `dt`.

But then `run_pipeline.py` applies a second EKF innovation gate using `vel * 1.0`.

So there are two different gates:

| Gate | Location | Position used | Time allowance |
|---|---|---|---|
| PF innovation gate | `temporal_searcher.py` | uncorrected homography | `velocity * dt` |
| EKF innovation gate | `run_pipeline.py` | look-ahead-corrected homography | `velocity * 1.0` |

This inconsistency is bad. It can allow the PF update but reject the EKF update, or reject one reason and report another.

### 3.3 Recommended fix

Create one shared innovation-threshold helper or at minimum make both gates use the same formula.

Recommended formula should include:

```text
base floor: 150 m
+ 3 * EKF position sigma
+ velocity * actual_dt
+ optional latency allowance
+ additional growth after consecutive rejected visual updates
```

Do **not** simply remove the innovation gate. That would allow catastrophic wrong homographies. The correct fix is a recovery-aware gate, not no gate.

---

## 4. Gate Semantics Are Confused

### 4.1 `gate_pass` currently means different things in different places

In `temporal_searcher.py`, the returned `gate_pass` near the bottom only checks visual quality:

```python
"gate_pass": (cshape > self.cfg.QUALITY_GATE_CSHAPE
              and n_inliers > self.cfg.QUALITY_GATE_INLIERS
              and homo_position is not None),
```

But in `run_pipeline.py`, `gate_pass` is overwritten after the EKF innovation gate:

```python
result["gate_pass"] = gate_pass
```

So `gate_pass` sometimes means:

- visual quality passed,
- EKF innovation passed,
- EKF update was actually applied,
- or all of the above depending on where you inspect it.

That is messy and dangerous.

### 4.2 Why this matters

Frame 126 is visually strong, but the final CSV says `gate_pass = 0`.

A human reader may interpret that as:

> "The visual localisation failed."

But the correct interpretation is:

> "The visual localisation succeeded, but the EKF innovation gate rejected it."

That is a completely different diagnosis.

### 4.3 Recommended fix

Add separate fields:

```text
visual_quality_pass
pf_innovation_gate_pass
ekf_innovation_gate_pass
ekf_update_applied
relocalization_candidate
visual_rejected_reason
pf_update_source
```

Then define `gate_pass` clearly as either:

```text
gate_pass = ekf_update_applied
```

or deprecate `gate_pass` and use explicit fields only.

For the thesis, this distinction is critical.

---

## 5. PF Update and EKF Update Are Inconsistent

### 5.1 The PF can update from one source while EKF rejects the final visual update

In `temporal_searcher.py`, if homography innovation is too large, it may fall back to tile-center updates:

```python
elif meta_result["verified"]:
    plausible = []
    ...
    if plausible:
        pf_update_source = "tile_center"
        measurements = [...]
```

In the late run, many frames have:

```text
pf_update_source = tile_center
visual_rejected_reason = innovation_too_large
r_used_sqrt = null
```

This means:

- the visual system found a strong meta-tile/homography,
- the PF did not always use the homography,
- the EKF rejected the visual update,
- the PF may still be nudged by coarse tile centers.

That is not clean fusion logic.

### 5.2 Tile-center fallback is too crude for recovery

A tile center is not a position measurement. At zoom 16 and 512 px tiles, the tile itself spans hundreds of meters. Updating the PF from tile centers is useful as rough search guidance, but it is too crude to recover a drifting estimator.

If homography is visually strong, the system should prefer the homography as a relocalization candidate instead of degrading to tile-center fallback.

### 5.3 Recommended fix

When visual quality is strong but innovation is large, classify the measurement as:

```text
relocalization_candidate = true
```

Then perform recovery logic, not ordinary rejection.

Recommended recovery conditions:

```text
CShape >= 0.70
inliers >= 200
semantic_conf >= 0.70
meta_tile_verified = true
verification_matches >= 100
innovation_too_large = true
consecutive innovation failures >= 3
homography positions coherent over last 2–3 frames
```

If these conditions hold, accept the visual measurement as a relocalization update.

---

## 6. Large-R Update Alone May Be Too Weak

A naive suggestion is:

```text
If recovery candidate, update EKF with large R = 200 m std.
```

But this may barely move the EKF if the EKF covariance is too small.

At frame 126:

```text
EKF pos sigma ≈ 42.9 m
innovation ≈ 337.1 m
```

If `R = 200²`, then the Kalman gain is roughly:

```text
K ≈ P / (P + R)
  ≈ 42.9² / (42.9² + 200²)
  ≈ 1840 / 41840
  ≈ 0.044
```

That would move the EKF only about:

```text
0.044 * 337 m ≈ 15 m
```

That is not enough. The EKF would still be very wrong.

### Recommended recovery update

A real recovery mode needs one of these:

### Option A — covariance inflation before recovery update

If a high-confidence relocalization candidate persists for `N` frames:

```python
ekf.P[8, 8] = max(ekf.P[8, 8], RELOCALIZATION_PRIOR_STD_M**2)
ekf.P[9, 9] = max(ekf.P[9, 9], RELOCALIZATION_PRIOR_STD_M**2)
ekf.update_position(homo_lat, homo_lon, R_pos_m2=RELOCALIZATION_R_M**2)
```

Possible values:

```text
RELOCALIZATION_PRIOR_STD_M = 150–250 m
RELOCALIZATION_R_M = 80–150 m
```

### Option B — controlled position reset

If the visual evidence is very strong for multiple frames, directly recenter the nominal EKF position and inflate covariance. This is more aggressive and should be clearly labelled as relocalization.

### Option C — two-stage correction

First update with moderate R, then if innovation remains high but coherent for the next frame, perform covariance inflation + update.

**Recommendation:** Start with Option A. It is safer than hard reset but strong enough to recover.

---

## 7. Homography Quality Scoring Is Still Incomplete

This is Phase B work, but it must be kept in mind.

### 7.1 Winner selection ignores reprojection error

In `visual_measurement.py`, `_select_homography_winner()` scores each branch as:

```python
return n * cs * bonus
```

This ignores `reproj_median`, even though the code already computes it.

Bad consequence:

> A homography with many inliers but poor reprojection consistency can beat a homography with fewer but much cleaner inliers.

Recommended score:

```python
score = inliers * CShape * convexity_bonus / (1.0 + reproj_median)
```

Also add hard rejection for excessive reprojection error.

### 7.2 No inlier-ratio check

The code checks inlier count, but not inlier ratio.

A homography with:

```text
500 raw matches, 25 inliers
```

has only 5% support. That may be bad even though it passes `QUALITY_GATE_INLIERS = 20`.

Add:

```text
MIN_INLIER_RATIO = 0.10 or 0.15
```

### 7.3 CShape can be high for a wrong but geometrically plausible homography

`compute_shape_confidence()` evaluates projected quadrilateral shape. This catches degenerate shapes, but it does not prove the projected map location is correct.

So CShape must be combined with:

- reprojection error,
- inlier ratio,
- semantic confidence,
- meta-tile verification,
- temporal consistency.

Do not rely on CShape alone.

---

## 8. Measurement Cascade Is Still Suspicious

### 8.1 `nadir_corrected` is always first

In `temporal_searcher.py`, `_build_cascade()` always returns:

```python
return ["nadir_corrected"] + base
```

where base is:

```python
["trimmed_centroid", "inlier_centroid", "weighted_centroid", "projected_center"]
```

This means the pipeline always tries `nadir_corrected` first.

### 8.2 `nadir_corrected` uses resized/rotated query width

In `visual_measurement.py`, `nadir_corrected` computes:

```python
f_px_approx = query_w / (2 * math.tan(math.radians(35)))
```

But `query_w` is the resized rotated image width, not necessarily the original camera width. That means the focal approximation can be wrong.

### 8.3 There may be double correction

The measurement extraction has `nadir_corrected`, then `run_pipeline.py` applies a separate `LOOKAHEAD_M = 110 m` correction.

So the actual position may be corrected twice:

1. inside `nadir_corrected`, using pitch/roll and approximate FOV;
2. outside in `run_pipeline.py`, using a 110 m heading look-ahead correction.

This might be okay if calibrated, but right now it is not clearly justified.

### Recommended fix

Phase B should test cascade alternatives:

```text
A. projected_center + external LOOKAHEAD only
B. nadir_corrected + no external LOOKAHEAD
C. current method
```

Do not assume current cascade is optimal.

---

## 9. Meta-Tile Construction Can Create Bad Geometry

### 9.1 Current top-K selection

`meta_tile_builder.py` uses:

```python
top_k = second_pass_results[:self.cfg.METATILE_TOP_K]
```

This chooses the top 3 tiles by match count.

Problem:

> The top 3 tiles may be spatially irregular. The canvas can contain black gaps. Then projected-center may fall into a black area and the cascade may fall back to centroid methods.

### 9.2 Why centroid fallback is dangerous

Centroid methods estimate the center of matched features, not the drone camera ground point.

If most matched features lie on a road, forest edge, or building cluster, the centroid is just the center of that feature cluster. That is not necessarily where the UAV is.

### Recommended fix

Build a consistent 3×3 patch around the best tile rather than arbitrary top-3 scattered tiles.

This gives:

- rectangular geometry,
- fewer black gaps,
- simpler pixel-to-lat/lon mapping,
- more stable homography projection.

---

## 10. Particle Filter Issues

### 10.1 PF measurement noise is huge

`config.py` sets:

```python
MEASUREMENT_NOISE_POSITION_M = 500.0
```

This means visual updates barely sharpen the particle filter. The comment says this is because domain shift makes tile matches unreliable.

That might be acceptable for search-region guidance, but then do not expect the PF to recover precise position. It is a search-guide, not a strong estimator.

### 10.2 PF update source must be logged and interpreted carefully

Late frames show mixed update sources:

```text
homography
tile_center
none
```

When `pf_update_source = tile_center`, the PF is not being updated with the actual homography position. It is being nudged toward coarse tile centers.

That should be shown in diagnostics.

### 10.3 Search radius cap is good but may mask divergence

`config.py` sets:

```python
MAX_TEMPORAL_SEARCH_RADIUS_M = 1500.0
```

This prevents insane search radius blow-up. Good.

But if the system is drifting and search radius is capped, it may fail to search the true area if the EKF/PF is already wrong. This is another reason recovery logic matters.

---

## 11. Trace and Analysis Tools Are Currently Misleading

### 11.1 `pipeline_trace.pdf` shows missing query images

The notebook says `pipeline_data` exists, but for selected frames it reports:

```text
[Step 1] query.jpg not found — re-run with SAVE_PIPELINE_TRACE=True
```

This means either:

- the runtime did not save the expected image files,
- the notebook expects different filenames,
- the PDF was generated from a partial/inconsistent trace folder,
- or trace writing failed silently.

The trace cannot be trusted as a full A-to-Z visual explanation until this is fixed.

### 11.2 Trace JSON does not expose enough gate detail

`_build_trace_json()` currently includes:

```python
"gate_pass": bool(result.get("gate_pass", False))
```

but it does not include key new diagnostics:

```text
visual_innovation_m
max_visual_innovation_m
visual_rejected_reason
pf_update_source
search_radius_capped
visual_quality_pass
innovation_gate_pass
ekf_update_applied
relocalization_candidate
```

So the trace notebook cannot explain why a good visual frame was rejected.

### 11.3 Notebook wording is now wrong

The notebook says things like:

```text
[GATE FAIL] -- EKF coasts on IMU predict only
```

But after the new code, gate fail can mean:

- visual quality failed,
- PF innovation failed,
- EKF innovation failed,
- visual strong but rejected,
- relocalization candidate not accepted.

So that wording is too simplistic and now misleading.

### Recommended fix

Update trace export and notebook display to show:

```text
Visual quality: PASS/FAIL
PF innovation gate: PASS/FAIL
EKF innovation gate: PASS/FAIL
EKF update applied: YES/NO
Rejected reason: ...
Relocalization candidate: YES/NO
PF update source: homography/tile_center/none
```

---

## 12. Runtime Is Too Slow for Real-Time Claims

The latest live analysis reports:

```text
Mean end-to-end latency ≈ 3076 ms
Median ≈ 3124 ms
P90 ≈ 3477 ms
Max ≈ 3950 ms
Mean drone displacement during inference ≈ 204.4 m
```

This means the UAV has moved about 200 m by the time the pipeline outputs its estimate.

That is not real-time navigation.

For thesis wording, call this:

```text
live prototype / online demonstration / near-real-time research implementation
```

Do not call it:

```text
real-time embedded navigation-ready system
```

unless runtime is reduced dramatically.

### Also: trace is enabled by default

`config.py` currently has:

```python
SAVE_PIPELINE_TRACE = True
```

The comment says it costs 80–150 ms/frame. It is useful for debugging, but it should not be default for final timing runs.

For clean performance evaluation:

```text
SAVE_PIPELINE_TRACE = False
SAVE_QUERY_FRAMES = False
SAVE_IMU_ROWS = False
SAVE_ANALYSIS_DATA = optional
SAVE_TIMING_DATA = True only when profiling
```

---

## 13. Recommended Fix Plan

Do not jump straight into full Phase B. Fix the gating/recovery failure first.

### Phase R0 — verify current behaviour

Before code changes:

1. Confirm the current `results.csv` has 36 `innovation_too_large` rejections.
2. Confirm frames 107–126 form a long rejection streak.
3. Confirm frame 126 is visually strong but EKF-rejected.
4. Confirm simulator GPS is not used in `ekf.update_position()` after initialization.

### Phase R1 — split gate semantics

Add result fields:

```text
visual_quality_pass
pf_innovation_gate_pass
ekf_innovation_gate_pass
ekf_update_applied
relocalization_candidate
relocalization_applied
visual_rejected_reason
```

Keep old `gate_pass` only for backward compatibility, but define it as:

```text
gate_pass = ekf_update_applied
```

### Phase R2 — unify innovation threshold calculation

Create a helper, e.g.:

```python
def compute_max_visual_innovation(pos_sigma_m, velocity_mps, dt_s,
                                  consecutive_rejections=0,
                                  floor_m=150.0):
    return max(
        floor_m,
        3.0 * pos_sigma_m
        + velocity_mps * max(dt_s, 1.0)
        + 50.0
        + consecutive_rejections * 25.0
    )
```

Use the same helper in:

- `TemporalSearcher` PF gating;
- `run_pipeline.py` EKF gating.

The exact constants should be evaluated, not blindly trusted.

### Phase R3 — add conservative relocalization mode

Add a state counter in `run_pipeline.py` or a small helper object:

```text
consecutive_visual_rejections
last_rejected_visual_positions
```

A frame becomes a relocalization candidate if:

```text
visual_quality_pass = true
meta_tile_verified = true
semantic_conf >= 0.70
CShape >= 0.70
inliers >= 200
verification_matches >= 100
visual_rejected_reason = innovation_too_large
```

Apply relocalization only if:

```text
consecutive_visual_rejections >= 3
AND recent rejected homography positions are mutually coherent
```

Coherence check example:

```text
The last 2–3 homography positions should move in a plausible direction and distance given velocity and heading.
```

### Phase R4 — recovery EKF update

Do not use weak large-R update alone. It may barely move the EKF.

Recommended recovery:

```python
# Before relocalization update:
ekf.P[8, 8] = max(ekf.P[8, 8], RELOCALIZATION_PRIOR_STD_M ** 2)
ekf.P[9, 9] = max(ekf.P[9, 9], RELOCALIZATION_PRIOR_STD_M ** 2)

ekf.update_position(homo_lat, homo_lon, R_pos_m2=RELOCALIZATION_R_M ** 2)
```

Suggested initial values:

```text
RELOCALIZATION_PRIOR_STD_M = 150–250 m
RELOCALIZATION_R_M = 80–150 m
```

These need tuning with the Odense run.

### Phase R5 — update trace outputs

Update `_build_trace_json()` and the trace notebook to include all gate/recovery diagnostics.

The trace should clearly say:

```text
Visual localisation succeeded, but EKF rejected due to innovation gate.
```

or:

```text
Relocalization candidate accepted after N consecutive failures.
```

### Phase R6 — rerun and evaluate

Compare current run vs fixed run:

| Metric | Current | Desired after fix |
|---|---:|---:|
| innovation rejections | 36 | lower |
| long rejection streak 107–126 | yes | broken/recovered |
| final frame error | 343.6 m | much lower |
| mean EKF error | 108.4 m | lower than corrected homography or at least close |
| median EKF error | 82.2 m | closer to corrected homography median 41.0 m |
| corrected homography accepted when strong | no | yes/relocalized |

Only after this should Phase B homography-quality changes begin.

---

## 14. Phase B Still Needed Later

After gating/recovery is fixed, Phase B should address:

1. Reprojection-error penalty in homography winner scoring.
2. Inlier-ratio threshold.
3. Projected-center vs nadir-corrected cascade testing.
4. Original camera width for FOV-based nadir correction.
5. Consistent 3×3 meta-tile patch instead of arbitrary top-3.
6. Scale sanity check in `compute_shape_confidence()`.
7. Runtime performance cleanup.

But do not mix all of this with recovery logic at once. That makes debugging impossible.

---

## 15. Concrete Tasks for Claude Code to Investigate

Claude Code should verify this review against the source and produce a final implementation plan.

### Required investigation

1. Confirm whether `ekf.update_position()` is only called from visual measurements after initialization.
2. Confirm line-by-line that GP8 simulator-GPS fallback is gone.
3. Confirm why frames 107–126 are rejected.
4. Confirm whether `gate_pass` is visual-quality-only in `TemporalSearcher` but final-EKF-update in `run_pipeline.py`.
5. Confirm whether PF update and EKF update use inconsistent innovation thresholds.
6. Confirm whether `vel * 1.0` in `run_pipeline.py` is too strict relative to actual frame `dt`.
7. Confirm whether trace JSON lacks the new diagnostics.
8. Confirm whether `query.jpg` saving and notebook expectations match.
9. Evaluate recovery logic options and choose a safe staged implementation.
10. Do not start full Phase B until recovery logic is tested.

### Expected output from Claude Code

Claude should produce:

1. Verified problem list.
2. Exact file/function/line references.
3. Proposed final fix plan.
4. Risk rating per change.
5. Validation plan.
6. Decision on whether to implement:
   - gate-splitting first,
   - recovery mode first,
   - threshold unification first,
   - trace diagnostics first.
7. Then ask for approval before execution.

---

## 16. Minimal Acceptance Criteria After Fix

A fixed version should satisfy:

- No simulator/GPS lat/lon enters EKF update after initialization.
- `gate_pass` no longer hides the reason for rejection.
- Strong visual measurements rejected by innovation gate are explicitly marked as `relocalization_candidate`.
- The system can recover from a long innovation-rejection streak.
- Frame 126-type cases no longer remain permanently rejected if visual evidence is consistently strong.
- Trace notebook explains accepted/rejected/relocalized visual measurements correctly.
- Corrected homography and EKF fusion are evaluated separately.
- EKF should not be worse than corrected homography over most of the run after recovery logic is implemented.

---

## 17. Short Final Diagnosis

The current pipeline is not garbage. The visual localisation has strong evidence of working. The estimator is the weak link.

The code now behaves like:

```text
I trust the EKF unless visual agrees with it.
```

But for GPS-denied recovery, it should behave more like:

```text
I trust the EKF during normal operation.
If visual disagrees once, be suspicious.
If visual disagrees repeatedly but is geometrically, semantically, and temporally consistent, treat it as relocalization and recover.
```

That is the core fix.
