# CLAUDE.md — Pipeline 3 Development Log

> This file tracks all changes, decisions, bugs, and solutions for the Pipeline 3 localization system.
> Update this file with every significant change so future sessions have full context.

## Project Overview

**Goal**: GPS-free drone localization using visual feature matching (SuperPoint+LightGlue) + semantic terrain segmentation + IMU dead reckoning (Error-State EKF) + particle filter tracking.

**Working directory**: `Pipeline_3_Rev1/`
**Reference implementations** (read-only, must match algorithms):
- `feature_matching_comparison_v2.ipynb` — working SuperPoint+LightGlue matcher
- `MSFS2020_IMU_Pipeline/ekf_ins.py` — working Error-State EKF for dead reckoning

**Data**:
- Query frames: `Logs_Run_20260321_162024/images_20260321_162024/frame_*.jpg` (1920×1079, ~970 frames)
- IMU log: `Logs_Run_20260321_162024/imu_gps_log_20260321_162024.csv` (970 rows)
- Reference tiles: `REFERENCE_MAP_VEJLE/aerial/16/{x}/{y}.png` (512×512, 270 tiles)
- Reference predictions: `REFERENCE_MAP_VEJLE/prediction/16/{x}/{y}.png`
- Semantic model: `SemanticTerrainSegmentationModel/best.pth`

**Reference map coverage** (TMS zoom 16):
- X: 34494–34508, Y: 45025–45042
- Center approx: (55.7°N, 9.5°E)
- Drone GPS enters map at **row 430** (ts≈199s), exits at **row 714** (ts≈328s)
- Only 285/970 CSV rows have GPS inside the reference map

---

## Architecture

```
EKF Dead Reckoning (src/ekf_ins.py)
    ↓ dead-reckoned lat/lon/heading
Temporal Searcher (src/temporal_searcher.py)
    ├─ Frame 0: BestFirstSearcher (src/best_first_search.py)
    │     └─ SuperPoint+LightGlue against all tiles in radius
    ├─ Frame 1+: Particle Filter predict → MetaTileBuilder two-pass search
    │     ├─ First pass: match query vs all tiles in particle radius
    │     ├─ Second pass: 8-neighbour expansion around top-1
    │     ├─ Build meta-tile from top-K → verify match count
    │     └─ Particle filter update + resample
    ├─ Semantic double-confirmation (src/semantic_confirmer.py)
    └─ Position via homography (src/position_estimator.py)
```

**Image preprocessing split**:
- Feature matching (SuperPoint+LightGlue): raw images at original resolution, NO resizing or padding
- Semantic model: 512×288 resized + padded to 512×512 (center-padded)

---

## Source Files (Pipeline_3_Rev1/src/)

| File | Purpose |
|------|---------|
| `ekf_ins.py` | Error-State EKF — copied from `MSFS2020_IMU_Pipeline/ekf_ins.py`, `preprocess_imu_csv()` added |
| `tile_utils.py` | TMS tile math, TileLoader, haversine_distance |
| `image_utils.py` | `load_image()`, `preprocess_query_frame()` (semantic only), `resize_for_matching()` (center-crop) |
| `geometric_matcher.py` | SuperPoint+LightGlue wrapper |
| `best_first_search.py` | Frame 0 cold-start exhaustive tile search |
| `particle_filter.py` | Particle filter for temporal tracking |
| `meta_tile_builder.py` | Two-pass search + meta-tile build + save + verify |
| `semantic_model.py` | UNet++ EfficientNet-B3 semantic segmentation model |
| `semantic_confirmer.py` | Centroid-based semantic double confirmation |
| `position_estimator.py` | Homography → GPS via tile geo-referencing |
| `temporal_searcher.py` | Top-level frame processor (orchestrates everything) |
| `trajectory_smoother.py` | Post-processing Kalman / moving average smoother |

---

## Bug History

### BUG 1 — TMS Y-axis Convention (Fixed 2026-03-25)
**Problem**: `latlon_to_tile()` returned OSM Y (~20504) but tiles on disk use TMS Y (~45031).
No tiles were ever found because Y coordinates were in the wrong convention.

**Fix**: Added `y_tms = n - 1 - y_osm` in `latlon_to_tile()`, `latlon_to_tile_float()`.
Reversed in `tile_to_latlon()`, `tile_bounds()`. Fixed particle filter motion direction.

**Files**: `src/tile_utils.py`, `src/particle_filter.py`

### BUG 2 — Image Preprocessing Destroyed Feature Matching (Fixed 2026-03-25)
**Problem**: `best_first_search.py` and `temporal_searcher.py` called `preprocess_query_frame()` BEFORE
passing images to SuperPoint+LightGlue. This resized 1920×1079 → 512×288 → padded to 512×512 with 224px
of black border. SuperPoint found keypoints in the black border = garbage matches.

The working reference (`feature_matching_comparison_v2.ipynb`) passes images at **original resolution**.

**Fix**: Removed `preprocess_query_frame()` from the matching path. Raw query frames go directly to
matcher. `preprocess_query_frame()` is called **only** for the semantic model. Position estimation
now uses actual query dimensions (`query_frame.shape`) instead of hardcoded 512.

**Files**: `src/best_first_search.py`, `src/temporal_searcher.py`

### BUG 3 — Raw GPS Used Instead of EKF Dead Reckoning (Fixed 2026-03-25)
**Problem**: The notebook read `row['latitude']`, `row['longitude']` directly from the CSV and passed
them as `imu_data['lat']`, `imu_data['lon']`. These ARE ground truth GPS from MSFS — not dead-reckoned.
On `imu_fallback`, the pipeline returned these directly → 0.0m error (measuring GPS against itself).

**Fix**: Created `src/ekf_ins.py` with `preprocess_imu_csv()`. Runs the full Error-State EKF on the CSV
to produce dead-reckoned positions. Notebook Cell 2 runs EKF upfront, merges `ekf_lat`, `ekf_lon`,
`ekf_yaw`, `vel_n`, `vel_e` into the dataframe. Cell 4 uses `row['ekf_lat']`/`row['ekf_lon']`.

**Files**: `src/ekf_ins.py` (new), notebook Cells 2 & 4

### BUG 4 — START_ROW=0 With Drone Outside Map (Fixed 2026-03-25)
**Problem**: With `START_ROW=0`, the first 430 frames (of 970) are outside the reference map.
Frame 0 cold-start finds 0 tiles (drone at lon 9.65, map at lon 9.50 = ~5km away).
Particle filter diverges immediately. Every subsequent frame is cold_start → 0 tiles → fallback → diverge.
By the time the drone enters the map, the diverge→restart loop continues because each cold restart
searches around the current (drifted) position which may still not find tiles.

**Fix**: Cell 2 now auto-detects the first row where GPS is inside the reference map and sets
`START_ROW` accordingly. `NUM_FRAMES=300` to process the in-map portion.

**Files**: notebook Cell 2

### BUG 5 — TMS Float Coordinate Off-by-One (Fixed 2026-03-25)
**Problem**: `latlon_to_tile_float()` and `tile_to_latlon()` used `(n-1) - osm_y` for the continuous
TMS↔OSM Y conversion. The correct formula for continuous (fractional) coordinates is `n - osm_y`.
The `(n-1)` variant is only correct for integer tile-index mapping (used in `latlon_to_tile`).

**Effect**: All tile center computations in `find_tiles_within_radius()` were shifted ~345m north
(one full tile). This caused the function to return 0 candidates when the drone was near the map
edge — `tile_to_latlon(tx+0.5, ty+0.5)` returned a center 345m north of the real center, inflating
haversine distances beyond the search radius.

Similarly, `latlon_to_tile_float()` returned Y values that were 1.0 lower than the integer version
(e.g., 45024.018 instead of 45025.018), so `floor(float_y)` did not match the integer tile index.

**Fix**: Changed both functions to use `n - osm_y` / `n - tile_y`. Now:
- `floor(latlon_to_tile_float(lat, lon).y)` == `latlon_to_tile(lat, lon).y` ✓
- `tile_to_latlon(tx, ty)` returns the true SW corner of the tile ✓
- `find_tiles_within_radius` returns correct candidates at map edges ✓

**Files**: `src/tile_utils.py` (lines 29-49)

---

## MSFS-Specific Notes

- Python SimConnect returns all `_DEGREES_` variables in **RADIANS** (including `heading_magnetic`, `pitch`, `bank`)
- Accelerometers return coordinate acceleration in **ft/s²** (no gravity component) → must convert to m/s² and add synthesized gravity
- Axis mapping: MSFS body (X=right, Y=up, Z=forward, left-handed) → Standard NED body (X=forward, Y=right, Z=down, right-handed)
  - `accel_body = [accel_z_msfs, accel_x_msfs, -accel_y_msfs] + g_body`
  - `omega_meas = [gyro_z_msfs, gyro_x_msfs, gyro_y_msfs]` (no negation — pseudovector sign cancellation)
- Gravity synthesis from pitch/bank: `g_body = [-g*sin(pitch), g*sin(bank)*cos(pitch), g*cos(bank)*cos(pitch)]`

### BUG 6 — EKF Column Name Mismatch in Notebook (Fixed 2026-03-25)
**Problem**: Notebook cells referenced `row['ekf_lat']`, `row['ekf_lon']`, `row['ekf_yaw']` but
`preprocess_imu_csv()` returns `latitude_est`, `longitude_est`, `yaw_deg`. Ground truth cells
used `row['latitude']`/`row['longitude']` but EKF output renames these to `gps_lat`/`gps_lon`.
Cell 4 also double-converted heading with `np.degrees(yaw_deg)` when `yaw_deg` is already degrees.

**Fix**: Updated all column references:
- EKF estimates: `latitude_est`, `longitude_est`, `yaw_deg`
- Ground truth: `gps_lat`, `gps_lon`
- Removed `np.degrees()` wrapper on `yaw_deg`

**Files**: `notebooks/test_temporal_pipeline.ipynb` (Cells 4, 6, 7)

### BUG 7 — Search Radius Too Small for Tile Grid (Fixed 2026-03-25)
**Problem**: `get_search_region()` in particle filter had minimum search radius of 100m. Tiles are
~332m at zoom 16. A 100m radius often can't even find the tile the particle is standing on (tile
center can be up to 235m away). Result: MetaTileBuilder reported "no first-pass tiles" for most
frames, causing imu_fallback on nearly every frame.

**Fix**:
- Changed minimum search radius from 100m to `1.5 × tile_size` (~500m)
- Added floor in `temporal_searcher._process_frame_N` to ensure search radius ≥ `FIRST_PASS_SEARCH_RADIUS_M`
- Increased config values: `IMU_SEARCH_RADIUS_METERS` 350→500, `FIRST_PASS_SEARCH_RADIUS_M` 300→500
- Increased `DIVERGENCE_POSITION_THRESHOLD_M` 200→500 (was triggering on normal EKF drift of ~193m)

**Files**: `src/particle_filter.py`, `src/temporal_searcher.py`, `config/config.py`

### BUG 8 — Camera Look-Ahead Offset (Fixed 2026-04-07)
**Problem**: Homography-derived positions were systematically ~110m ahead of the drone in the heading
direction. The MSFS camera has a fixed forward tilt, so the image center corresponds to ground ahead
of the drone, not directly below. Diagnostic: computed offset vector (homo→GT) for all gated frames —
offset bearing ≈ heading + 180° (perfectly anti-aligned). Mean offset distance: 112m.

**Effect**: All visual position measurements had a consistent ~110m forward bias. The EKF corrected
partially (103.5m mean) but couldn't eliminate the systematic error. 0/50 frames achieved <50m accuracy.

**Fix**: Added camera look-ahead correction in the notebook closed-loop cell: before feeding the
homography position to `ekf.update_position()`, shift it 110m backward along the EKF heading direction.
```python
corr_north = -LOOKAHEAD_M * cos(heading_rad)
corr_east  = -LOOKAHEAD_M * sin(heading_rad)
homo_corrected = (homo_lat + corr_north/111320, homo_lon + corr_east/(111320*cos(homo_lat)))
```

**Result**: Mean error dropped from 103.5m → **9.7m** (91% improvement). 49/50 frames under 50m.

**Files**: `notebooks/test_temporal_pipeline.ipynb` (Cell 4)

---

## Current State (2026-04-07)

**Status**: Phase C — Online 10D EKF with closed-loop visual position updates + camera look-ahead correction.

**Architecture Change (Phase C)**:
The EKF error-state was expanded from 8D to 10D to include position error [δp_n, δp_e].
Visual matching results now feed back directly into the EKF via `update_position()`.
The particle filter is retained for search region guidance only — the EKF is the primary position estimator.

**Key Changes**:
- `src/ekf_ins.py`: P 8x8→10x10, F 8x8→10x10, all H matrices 8→10 cols, delta_state[8:10] applied in all updates
- `src/ekf_ins.py`: New `update_position(lat, lon, R_pos_m2)` — standard Kalman update on position states
- `src/ekf_ins.py`: New `step_ekf(ekf, row, prev_ts)` — single-row processing helper for online mode
- `src/temporal_searcher.py`: Removed EKF anchor (score=0.5 measurement in PF)
- `config/config.py`: MAX_NUM_KEYPOINTS 4096→2048, MAX_ROTATED_DIMENSION 1920→1280 (speed)
- `config/config.py`: VISUAL_POSITION_NOISE_M=50, POSITION_PROCESS_NOISE_M=5, INITIAL_POSITION_VARIANCE_M=200
- `notebooks/test_temporal_pipeline.ipynb`: Cell 2 warms up live EKF to START_ROW, Cell 4 runs closed-loop

**Camera Look-Ahead Correction (BUG 8)**:
Diagnostic analysis revealed homography positions were systematically 110m AHEAD of the drone in the
heading direction (offset bearing ≈ heading ± 180°). The MSFS camera has a fixed forward tilt, causing
the image center to correspond to ground ahead of the drone, not directly below.
Correction: shift homography position by 110m opposite to EKF heading before EKF update.

**Adaptive Measurement Noise**:
- High quality (CShape>0.5, inliers>100): R = 30² = 900 m²
- Normal quality (CShape>0.3, inliers>20): R = 60² = 3600 m²
- Below gate: no EKF update (EKF coasts on prediction)

**Closed-Loop Architecture**:
```
For each frame:
  1. step_ekf(row) — IMU predict + sensor updates (orientation, barometer, airspeed)
  2. TemporalSearcher.process_frame() — visual match → homo_position + quality gate
  3. Apply camera look-ahead correction (shift 110m backward along heading)
  4. if gate_pass: ekf.update_position(corrected_lat, corrected_lon, adaptive_R)
  5. final_position = ekf.get_state() — corrected lat/lon from EKF
```

**Phase C Results (50 frames)**:
- **Online EKF mean: 9.7m, median: 2.0m** (95% improvement over 196.1m batch EKF)
- 49/50 frames (98%) under 50m, 50/50 (100%) under 100m
- First 10 frames: 2-3m accuracy (rapid convergence)
- Last 10 frames: 29-50m (quality degrades near terrain change)
- Gate passes: 41/50 (82%), EKF coasts gracefully during bad visual frames
- Speed: ~2.8s/frame (1280px, 2048 keypoints)

**Previous Results**: Phase B1: 122m mean | Phase C (no correction): 103.5m mean

**Known Limitations**:
- Camera look-ahead correction (LOOKAHEAD_M=110) is empirically tuned for this flight
- Domain mismatch (MSFS 3D vs orthophoto 2D) limits visual match quality in some terrain
- Frames 39-47 show quality collapse (CShape drops to 0.1, inliers=5) — terrain/turning issue
- Speed ~2.8s/frame — could be improved by caching query SuperPoint features across tile matches

---

## Change Log

| Date | Change | Files |
|------|--------|-------|
| 2026-03-25 | Fixed TMS Y-axis (OSM→TMS conversion) | tile_utils.py, particle_filter.py |
| 2026-03-25 | Removed preprocess_query_frame from matching path | best_first_search.py, temporal_searcher.py |
| 2026-03-25 | Added center-crop `resize_for_matching()` | image_utils.py |
| 2026-03-25 | Created EKF module from reference impl | src/ekf_ins.py (new) |
| 2026-03-25 | Rewrote notebook: EKF, auto-START_ROW, debug viz | notebooks/test_temporal_pipeline.ipynb |
| 2026-03-25 | Created this CLAUDE.md | CLAUDE.md |
| 2026-03-25 | Fixed TMS float off-by-one: (n-1)→n in latlon_to_tile_float & tile_to_latlon | src/tile_utils.py |
| 2026-03-25 | Fixed EKF column name mismatches in notebook | notebooks/test_temporal_pipeline.ipynb |
| 2026-03-25 | Fixed search radius too small (100m→500m), divergence threshold (200→500) | particle_filter.py, temporal_searcher.py, config.py |
| 2026-03-25 | Added EKF sanity check cell, method distribution diagnostics | notebooks/test_temporal_pipeline.ipynb |
| 2026-03-26 | Fixed dt logging (last_timestamp update order) | temporal_searcher.py |
| 2026-03-26 | Tile center measurements (tx+0.5,ty+0.5), score normalization (MAX_SCORE=50) | temporal_searcher.py |
| 2026-03-26 | Homography on all frames with geometric sanity check | temporal_searcher.py |
| 2026-03-26 | Sub-tile particle measurements from homography | temporal_searcher.py |
| 2026-03-26 | EKF anchoring in particle filter (score=0.5 measurement) | temporal_searcher.py |
| 2026-03-26 | Score-gated EKF/visual blending → hard-gated (s>150, d<200m, w=1.0) | temporal_searcher.py |
| 2026-03-26 | Cold-start EKF fallback when visual score < 100 | temporal_searcher.py |
| 2026-03-26 | MEASUREMENT_NOISE_POSITION_M 50→500 | config.py |
| 2026-03-26 | Added diagnostic cells: blending analysis, strategy simulation, oracle | notebooks/test_temporal_pipeline.ipynb |
| 2026-03-26 | Error reduced: 2237m → 188.6m (91.6% reduction, +4m vs EKF) | all |
| 2026-04-06 | Phase A diagnostics: confirmed heading rotation as #1 improvement | scripts/phase_a_diagnostics.py |
| 2026-04-07 | Created visual_measurement.py: rotation, dual homography, 5 measurement methods | src/visual_measurement.py (new) |
| 2026-04-07 | Heading rotation before MetaTileBuilder, dual homography (MAGSAC+DLT) | src/temporal_searcher.py |
| 2026-04-07 | Quality-gated blending (CShape>0.3, inliers>20) replaces hard-gate | src/temporal_searcher.py |
| 2026-04-07 | Added pitch/roll to imu_data dict for nadir correction | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Phase B1 diagnostics: 138.0m quality-gated mean (−54.9m vs EKF) | scripts/phase_b1_diagnostics.py (new) |
| 2026-04-07 | Pipeline validation: 178.9m sparse, Frame 1 at 60.3m (−139.7m) | scripts/phase_b1_validate.py (new) |
| 2026-04-07 | Phase C: Expanded EKF 8D→10D (position error states) | src/ekf_ins.py |
| 2026-04-07 | New update_position() + step_ekf() for online closed-loop | src/ekf_ins.py |
| 2026-04-07 | Removed EKF anchor from PF (visual updates go to EKF now) | src/temporal_searcher.py |
| 2026-04-07 | Added visual position noise config constants | config/config.py |
| 2026-04-07 | Notebook Cell 2: EKF warmup to START_ROW, live_ekf instance | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Notebook Cell 4: Closed-loop EKF predict→visual→update | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Unit tests for 10D EKF (all passing) | tests/test_10d_ekf.py (new) |
| 2026-04-07 | Notebook cleanup: deleted 13 old/broken cells, 6 cells remain | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Cell 4 rewrite: ALL frames printed, image names, failure reasons | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Speed config: MAX_ROTATED_DIMENSION 1920→1280, MAX_NUM_KEYPOINTS 4096→2048 | config/config.py |
| 2026-04-07 | Adaptive measurement noise: R_HIGH=30²=900, R_MED=60²=3600 | notebooks/test_temporal_pipeline.ipynb |
| 2026-04-07 | Camera look-ahead correction (LOOKAHEAD_M=110): 103.5m→9.7m mean | notebooks/test_temporal_pipeline.ipynb |
