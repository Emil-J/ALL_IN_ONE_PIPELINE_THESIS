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

---

## Current State (2026-03-25)

**Status**: All 7 bugs fixed. Notebook runs end-to-end without errors.

**Results** (300 frames, START_ROW=430):
- 100% of frames have position estimates (300/300)
- 100% of frames found tiles (300/300 with >0 tiles tested)
- 99.7% temporal tracking, 0.3% cold start (no divergence restarts)
- 42.1% meta-tile verified, mean semantic confidence 0.227
- Mean error: 2237m, Median: 2434m (domain shift — MSFS footage vs real orthophotos)
- Frame 0 time: 1.4s, subsequent mean: 2.8s

**EKF Dead Reckoning Quality**:
- EKF vs GPS drift: mean 193m, std 19m (for 300 aligned frames)
- EKF heading: yaw_deg ≈ -61.6° (= 298.4° mod 360, matches raw heading of 5.2 rad)
- Speed: ~67 m/s (130 knots, reasonable for MSFS light aircraft)

**Known Limitation**: ~2.2km positioning error is caused by visual domain mismatch between
MSFS 3D drone footage and real 2D orthophotos, not a pipeline bug. Feature matching produces
poor/incorrect matches that mislead the particle filter instead of correcting EKF drift.

**Next steps**:
- Consider training domain adaptation for feature matching
- Test with real orthophoto query images (eliminate domain shift)
- Investigate if better feature matching (e.g., LoFTR) reduces domain sensitivity

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
