# Next Steps: Phase B and Beyond

> Written 2026-05-09. **Phase A is complete.** Phase B (homography quality root-cause fixes) is the immediate next step — see below. Phases C–H follow after Phase B produces stable live results.
> Key files: `runtime/run_pipeline.py`, `src/temporal_searcher.py`, `src/visual_measurement.py`,
> `src/meta_tile_builder.py`, `src/particle_filter.py`, `config/config.py`.

---

## Phase B — Homography Quality Root-Cause Fixes

**Motivation**: Phase A safety gates prevent the EKF and PF from accepting visually implausible measurements, but they do not fix why the measurements are wrong in the first place. Phase B addresses the root causes so that more measurements pass the gate correctly, rather than being silently rejected.

All 6 fixes are in `src/visual_measurement.py`, `src/temporal_searcher.py`, and `src/meta_tile_builder.py`. Apply and re-run live 120-frame test after each fix to verify no gate-pass rate regression.

### B1 — Fix nadir_corrected FOV to use original camera dimensions
**File**: `src/visual_measurement.py` (function `extract_visual_measurements`, ~line 456)

The focal length is computed from the resized-rotated query width (~1280px after B4's canvas expansion). The correct reference is the original 1920px frame width. The error makes the nadir correction ~50% wrong in scale.

Add parameter `original_query_w: int = 1920` to `extract_visual_measurements()`. Replace:
```python
f_px_approx = query_w / (2 * math.tan(math.radians(35)))
```
with:
```python
f_px_approx = original_query_w / (2 * math.tan(math.radians(35)))
```
Pass `original_query_w = query_frame.shape[1]` (before rotation/resize) from `temporal_searcher._process_frame_N()` and `_process_frame_0()`.

### B2 — Add reprojection error penalty to winner score
**File**: `src/visual_measurement.py:303–307` (`_select_homography_winner`)

`reproj_median` is computed but not used in the scoring. A branch with 30 inliers and 15px reprojection error beats 20 inliers at 1px. Replace:
```python
return branch["inliers"] * branch["CShape"] * (1.5 if branch["convex"] else 1.0)
```
with:
```python
reproj_penalty = 1.0 + branch["reproj_median"]
return branch["inliers"] * branch["CShape"] * (1.5 if branch["convex"] else 1.0) / reproj_penalty
```
Also add hard rejection if `reproj_median > MAX_REPROJ_PX`. Add `MAX_REPROJ_PX = 8.0` to `config/config.py`.

### B3 — Add minimum inlier ratio check
**File**: `src/visual_measurement.py:286–315` (RANSAC branch evaluation)

100 raw matches → 5 inliers (5%) means RANSAC found mostly noise; the homography is untrustworthy even if it nominally passes the inlier count gate. Compute `inlier_ratio = inliers / n_correspondences` after RANSAC. If `inlier_ratio < MIN_INLIER_RATIO`, mark branch invalid. Add `MIN_INLIER_RATIO = 0.10` to `config/config.py`.

### B4 — Change cascade order: projected_center first
**File**: `src/temporal_searcher.py:540–552` (`_build_cascade`)

`nadir_corrected` is always tried first, propagating a wrong FOV scale (root cause 3) as the primary position. `projected_center` projects the image center through H directly — no camera model needed. The look-ahead correction in `run_pipeline.py` already removes the systematic forward bias.

Replace current cascade with:
```python
base = ["projected_center", "trimmed_centroid", "inlier_centroid", "weighted_centroid"]
pitch_large = abs(pitch_rad) > 0.10   # ~6 degrees
roll_large  = abs(roll_rad) > 0.10
if pitch_large or roll_large:
    return ["nadir_corrected"] + base
return base
```
`nadir_corrected` only attempted when attitude angles are large enough to justify a camera model correction, and only after B1 fixes the scale.

### B5 — Use consistent 3×3 meta-tile patch instead of arbitrary top-K
**File**: `src/meta_tile_builder.py:277–351` (`run()` orchestration)

The top-3 tiles by match count can be geometrically scattered (e.g. three corners of a 3×3 grid), producing a meta-tile with black gaps. After second pass identifies the best tile, build a fixed 3×3 grid patch around it instead of arbitrary top-K:
```python
best_tx, best_ty = top1_tx, top1_ty
grid_tiles = []
for dx in range(-1, 2):
    for dy in range(-1, 2):
        tx, ty = best_tx + dx, best_ty + dy
        if self.tiles.exists(tx, ty):
            score = match_scores.get((tx, ty), 0)
            grid_tiles.append((tx, ty, score))
```
**Config**: Replace `METATILE_TOP_K = 3` with `METATILE_PATCH_SIZE = 3` (meaning ±1 neighbourhood radius, not "top K by score").

### B6 — Add scale consistency check on homography projected quad
**File**: `src/visual_measurement.py` (`compute_shape_confidence()`, ~line 73–159)

After computing `proj_area`, verify the projected quad covers a plausible fraction of the meta-tile for the expected drone altitude and FOV. At zoom 16 / 300m AGL / ~70° FOV, the footprint is ~615×385px on a 512–1536px meta-tile canvas. Add:
```python
"scale_ok": 0.01 * expected_area < proj_area < 10 * expected_area
```
If `scale_ok == False`, set `CShape = 0` to treat the branch as degenerate.

---

## Phase C — Measurement Method Diagnostics and A/B Testing

**Motivation**: After Phase B fixes the cascade order and FOV calculation, it is unknown which measurement method performs best across flight conditions. Phase C provides the logging and test harness to answer this empirically.

### C1 — Log selected measurement method per frame
Add `selected_measurement_method` to `results.csv`. This reveals whether the current cascade (projected_center first after B4) is actually selecting the correct method in practice.

**File**: `src/temporal_searcher.py` — propagate `mname` when cascade selects a method; include it in the return dict. `run_pipeline.py` — add to `RESULT_COLUMNS` and `result_row`.

### C2 — Log per-method position estimates for gate-pass frames
For gate-pass frames, log what each cascade method would have produced:
- `projected_center_lat`, `projected_center_lon`
- `nadir_corrected_lat`, `nadir_corrected_lon`

Compare against `gps_lat`/`gps_lon` offline. This lets you empirically determine which method is most accurate for this camera/flight combination without multiple live runs.

**File**: `src/temporal_searcher.py` — add measurement dump to trace_data; `run_pipeline.py` — optionally add selected columns to analysis_extras.csv.

### C3 — Log raw vs corrected homography error offline
Using `gps_lat`/`gps_lon` (logging only), compute:
- `homo_raw_error_m` = haversine(homo_lat_raw, homo_lon_raw, gps_lat, gps_lon)
- `homo_corrected_error_m` = haversine(homo_corrected_lat, homo_corrected_lon, gps_lat, gps_lon)

Add these to `analysis_extras.csv` (not `results.csv` — evaluation-only columns).

Acceptance criteria: `homo_corrected_error_m` < `homo_raw_error_m` for ≥ 70% of gate-pass frames. If not, re-examine the look-ahead correction value (LOOKAHEAD_M = 110).

### C4 — Semantic hard gate (gated on diagnostic confirmation)
After C3 confirms semantic_conf correlates with error, add:
```python
MIN_SEMANTIC_CONF_FOR_UPDATE = 0.5  # config.py
if sem_conf < MIN_SEMANTIC_CONF_FOR_UPDATE:
    gate_pass = False
    visual_rejected_reason = "semantic_confidence_low"
```
**File**: `config/config.py`, `runtime/run_pipeline.py` (both paths).

**Risk**: Semantic model may classify terrain inconsistently. Do not enable until C3 data shows sem_conf < 0.5 reliably predicts high error.

### C5 — meta_verified=False hard reject
After diagnostics from C3, if meta_verified=False correlates with large error:
```python
if not meta_verified:
    gate_pass = False
    visual_rejected_reason = "meta_tile_not_verified"
```
Replace the current `r_used *= 2.0` soft penalty with a hard reject.

**File**: `runtime/run_pipeline.py` (both paths).

---

## Phase D — Meta-Tile Construction and Tile Ranking Improvements

**Motivation**: The first-pass tile ranking by raw match count is weak on homogeneous terrain. The meta-tile from arbitrary top-K can have black gaps and produce degenerate homographies. Phase D addresses the full search quality.

### D1 — Geometry-aware first-pass candidate reranking
After first-pass match counts, take the top-N candidates (N=10) and run a lightweight secondary check:

Option 1: Histogram intersection between query semantic map and reference prediction tiles (already available from semantic pre-filter). Sort surviving candidates by combined match_count + semantic_score.

Option 2: For top-5 candidates, compute MAGSAC homography quality (inlier count, CShape) and rerank.

**Files**: `src/meta_tile_builder.py:first_pass()` and `src/temporal_searcher.py`.

**Acceptance criteria**: Top-1 candidate from first pass is the geographically correct tile on > 80% of gate-pass frames (verifiable using gps_lat/gps_lon offline).

### D2 — Consistent 3×3 meta-tile patch (from Phase B5, if deferred)
If Phase B5 was not implemented, do it here. Build a fixed 3×3 grid around the second-pass winner instead of arbitrary top-K.

**File**: `src/meta_tile_builder.py:run()`.

### D3 — Verify meta-tile with geometric quality check
The current verification only counts matches (`verified = match_count >= 25`). This passes even when the matches are all false positives from repeated texture.

Replace with a verification homography:
```python
ver_H, ver_mask = cv2.findHomography(ver_src_pts, ver_dst_pts, cv2.USAC_MAGSAC, 8.0)
ver_inliers = ver_mask.sum()
ver_CShape = compute_shape_confidence(ver_H, query_w, query_h)["CShape"]
verified = (ver_inliers >= METATILE_VERIFY_MIN_INLIERS   # e.g. 20
            and ver_CShape > METATILE_VERIFY_MIN_CSHAPE)  # e.g. 0.4
```
**Files**: `src/meta_tile_builder.py:verify_meta_tile()`, `config/config.py`.

### D4 — Altitude-adaptive look-ahead correction
The current `LOOKAHEAD_M = 110` is fixed for the Odense test flight at ~300m AGL. For different flight profiles, the correction should scale with altitude:

```python
effective_lookahead = LOOKAHEAD_M_PER_100M * (altitude_m / 100.0)
```
This requires altitude from EKF (`final["altitude"]`). Log `altitude_m` in existing results.csv, then calibrate the constant.

**Files**: `runtime/run_pipeline.py` (both paths), `config/config.py` (new constant `LOOKAHEAD_M_PER_100M`).

---

## Phase E — EKF and PF Integration Tightening

**Motivation**: EKF and PF currently run mostly independently. The EKF is the authoritative position estimator; the PF is used only for search region guidance. After Phase A+B, the search region may still drift because the PF only receives tile-center measurements (low precision) when the homography gate fails.

### E1 — Feed EKF position back into PF as high-precision anchor
When the EKF has low pos_sigma (< 100m), reseed PF particles around the EKF position:
```python
if ekf_pos_sigma < cfg.EKF_PF_RESEED_SIGMA_M:
    particle_filter.reseed_around(ekf_lat, ekf_lon, spread_m=ekf_pos_sigma * 2)
```
This prevents PF from drifting away from the EKF's well-calibrated position during visual gate failures.

**Files**: `src/temporal_searcher.py`, `src/particle_filter.py` (add `reseed_around()`), `config/config.py`.

### E2 — Dynamic look-ahead correction using EKF velocity
Instead of applying a fixed 110m backward correction, use the EKF's velocity estimate and processing delay:
```python
processing_delay_s = inference_time_s  # logged per frame
lookahead_m = np.sqrt(vel_n**2 + vel_e**2) * processing_delay_s
```
The look-ahead offset is then determined by actual velocity and actual processing latency. This self-calibrates across flight speeds.

**Files**: `runtime/run_pipeline.py` (both paths).

### E3 — Automatic cold-start reinitialisation on persistent pos_sigma growth
If `ekf_pos_sigma` exceeds a threshold for N consecutive frames, reinitialize the PF with a wide spread. Currently divergence detection is in the PF only:
```python
if self.particle_filter.check_divergence():
    self.frame_count = 0  # triggers cold start
```
This reinitialises the search but not the EKF. Add EKF reinit option or at least a warning alert.

**Files**: `src/temporal_searcher.py`, `src/ekf_ins.py`.

### E4 — EKF process noise tuning from empirical trajectory
The current `POSITION_PROCESS_NOISE_M = 5.0` m/√s was set empirically. After Phase A, the pos_sigma growth on gate-fail frames can be measured from results.csv. If sigma grows faster than expected, process noise should be reduced. If sigma grows too slowly (EKF over-confident), increase it.

Use the Phase C diagnostic data (gate-fail sequences) to tune this.

---

## Phase F — Robustness and Edge Case Handling

### F1 — Map boundary detection
If the EKF position drifts outside the reference map bounds, the search finds no tiles and the pipeline coasts on IMU indefinitely. Add early detection:
```python
in_map = (cfg.TILE_X_MIN <= tx <= cfg.TILE_X_MAX
          and cfg.TILE_Y_MIN <= ty <= cfg.TILE_Y_MAX)
if not in_map:
    logger.warning("EKF position outside reference map — coasting on IMU")
```
**Files**: `runtime/run_pipeline.py` (both paths), `config/config.py`.

### F2 — Featureless terrain detection and early skip
When the query frame has very few SuperPoint keypoints (< MIN_KEYPOINTS threshold), skip the full matching pipeline and coast on IMU immediately:
```python
n_kp = len(query_feats["keypoints"])
if n_kp < cfg.MIN_KEYPOINTS_FOR_MATCHING:
    return self._imu_fallback_result(...)
```
This saves ~500ms per frame and avoids garbage matches on featureless frames (open water, uniform farmland).

**Files**: `src/temporal_searcher.py`, `config/config.py` (add `MIN_KEYPOINTS_FOR_MATCHING = 50`).

### F3 — GPS-denied integrity final check
After Phase F, run a formal re-audit of all GPS/EKF touchpoints using the same GP1-GP14 classification scheme. Update `docs/GPS_DENIED_INTEGRITY_AUDIT.md`, `docs/BS_CHECK.md`, and `docs/pipeline_breakdown.tex`. Confirm all Phase A/B changes are reflected.

---

## Phase G — Production and Performance

### G1 — Record a new Odense replay log for file-mode evaluation
The current file-mode config points to `Logs_Run_20260321_162024` (Vejle/CPH), which does not match the active Odense reference map. Record a new replay log over Odense and update `config/config.py`:
```python
IMU_CSV_PATH = ALL_IN_ONE_ROOT / "Logs_Odense_XXXX" / "imu_gps_log_XXXX.csv"
QUERY_FRAMES_DIR = ALL_IN_ONE_ROOT / "Logs_Odense_XXXX" / "images_XXXX"
```
File mode is the clean evaluation mode for thesis headline numbers (no GP8 fallback, no live mode artifacts).

### G2 — SAVE_PIPELINE_TRACE default to False
`config/config.py:244`: `SAVE_PIPELINE_TRACE = True` adds 80–150ms overhead per frame (image encoding). Change default to False; document how to enable for debugging. Add `--trace` CLI flag.

**Files**: `config/config.py`, `runtime/run_pipeline.py` (CLI arg).

### G3 — Full evaluation run (file mode, ~285 in-map frames)
After G1, run:
```powershell
python Pipeline_3_Rev1/runtime/run_pipeline.py --source file --run-id eval_odense_01
```
Compute official headline numbers:
- Median error (m)
- Mean error (m)
- % frames within 50m, 100m, 150m
- Gate-pass rate
- Processing speed (fps)

These become the thesis-defensible numbers.

### G4 — Processing speed optimisation
Current throughput ~1 fps. Priority targets:
1. Cache SuperPoint query features across the frame (done for reference tiles; also done for query in meta_tile_builder)
2. Skip semantic pre-filter when candidate count ≤ `SEMANTIC_PREFILTER_TOP_K`
3. SAVE_PIPELINE_TRACE=False for evaluation runs (G2)
4. Profile whether LightGlue or homography is the bottleneck

---

## Phase H — Thesis Documentation

Run after Phase G produces clean evaluation numbers.

### H1 — Update thesis docs to reflect Phase A+B changes
Files to update (all in `Pipeline_3_Rev1/docs/`):
- `GPS_DENIED_INTEGRITY_AUDIT.md` — reflect GP8 removal as "REMOVED 2026-05-09"; update live-mode verdict
- `BS_CHECK.md` — update Q3, Q5-Q12 and bottom-line verdict for post-GP8 state
- `CALL_GRAPH.md` — remove GP8 branch from section 3g; update live/file delta table
- `CODEMAP.md` — remove GP8 note from run_pipeline.py row
- `CURRENT_BEHAVIOUR_BASELINE.md` — mark `live_020_Odense_f1` as pre-GP8-removal; add new baseline from G3
- `pipeline_breakdown.tex` — update GP8 subsection to "Removed"; update quality-gate prose

### H2 — Update Mermaid diagrams
Files in `Pipeline_3_Rev1/docs/Diagrams/`:
- `04_file_replay_pipeline.mmd` — remove "contrast with live mode GP8" annotation
- `05_ekf_visual_fusion.mmd` — remove GP8 dashed subgraph
- `36_vl_fusion_output.mmd` — remove GP8_AUX node
- `37_vl_gp8_disclosure.mmd` — convert to historical note with "REMOVED 2026-05-09"

### H3 — Update top-level README
Remove the GP8 soft-anchor bullet from "Honest About the Limitations". Replace with:
> "Live mode and file mode both coast on IMU dead reckoning when the visual gate fails; no GPS is consumed by the estimator after the initial geodetic prior at t₀."

### H4 — Update CLAUDE.md change log
Add entry: "2026-05-09 — Removed GP8 live-mode sim-GPS fallback. Added Phase A safety gates (innovation gate, PF homography gate, search radius cap, gyro axis fix). Added Phase B homography quality fixes (reprojection penalty, inlier ratio, cascade order, 3×3 meta-tile, FOV fix)."

---

## Summary Table

| Phase | Priority | Risk | When |
|---|---|---|---|
| C — Diagnostics | High | Low | After Phase A+B live run stable |
| D — Meta-tile quality | High | Medium | After Phase C confirms which methods work |
| E — EKF/PF tightening | Medium | Medium | After Phase D |
| F — Robustness | Medium | Low | After Phase E |
| G — Production eval | High | Low | After Phase F |
| H — Thesis docs | High | Low | After Phase G eval run |
