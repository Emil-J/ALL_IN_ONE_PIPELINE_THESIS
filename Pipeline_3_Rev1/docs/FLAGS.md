# Pipeline 3 — Config Flags Reference

## Output flags (`config/config.py`)

| Flag | Default | What it saves | Timing impact |
|------|---------|--------------|---------------|
| `SAVE_QUERY_FRAMES` | `False` | `flight_data/frame_NNNN.jpg` — raw query JPEG | ~10–20 ms/frame |
| `SAVE_IMU_ROWS` | `False` | `flight_data/frame_NNNN_imu.json` — raw IMU row | <1 ms/frame |
| `SAVE_ANALYSIS_DATA` | `False` | `px4_gps_input.csv` + `analysis_extras.csv` | <1 ms/frame |
| `SAVE_TIMING_DATA` | `False` | `timing_data.csv` — per-component ms breakdown | <0.1 ms/frame |
| `SAVE_PIPELINE_TRACE` | `False` | `pipeline_data/frame_NNNN/` — 7 files per frame | 80–150 ms/frame |

## Debug flags (notebook/dev use only)

| Flag | Default | Effect |
|------|---------|--------|
| `DEBUG_SAVE_METATILES` | `False` | Saves meta-tile PNGs to `outputs/metatiles/` (global, not per-run) |
| `ACCUMULATE_HISTORY` | `False` | Keeps every frame result in `searcher.history` (unbounded memory growth) |

---

## What each flag saves in detail

### `SAVE_QUERY_FRAMES`
```
outputs/runs/<run_id>/flight_data/frame_NNNN.jpg
```
The raw query JPEG exactly as captured/loaded before any processing.

### `SAVE_IMU_ROWS`
```
outputs/runs/<run_id>/flight_data/frame_NNNN_imu.json
```
The raw IMU/GPS CSV row at that frame index. NaN values become `null`.

### `SAVE_ANALYSIS_DATA`
```
outputs/runs/<run_id>/px4_gps_input.csv
outputs/runs/<run_id>/analysis_extras.csv
```
- **px4_gps_input.csv**: MAVLink GPS_INPUT (MSG 232) ready for PX4 integration. Fields: `lat/lon` (degE7), `alt` (m), `vn/ve/vd` (m/s), `yaw` (centidegrees), `horiz_accuracy` (m), `ignore_flags=0x0006`.
- **analysis_extras.csv**: `n_eff`, `particle_spread`, `homo_offset_north_m`, `homo_offset_east_m` (look-ahead residuals vs GPS GT). Used by `live_analysis.ipynb` Cell 13.

### `SAVE_TIMING_DATA`
```
outputs/runs/<run_id>/timing_data.csv
```
Per-frame component timing in milliseconds. Columns: `cold_search_ms`, `pf_predict_ms`, `semantic_ms`, `meta_tile_ms`, `homography_ms`, `pf_update_ms`, `total_ms`, `frame_capture_ts`, `gps_estimate_ts`. Used by `live_analysis.ipynb` Cell 14.

### `SAVE_PIPELINE_TRACE`
```
outputs/runs/<run_id>/pipeline_data/frame_NNNN/
  query.jpg            — raw query frame
  query_rotated.jpg    — heading-aligned rotated frame (as fed to matcher)
  semantic_mask.png    — colour-coded semantic segmentation (512×512)
  reference_tile.png   — meta-tile (temporal) or best tile (cold start)
  matches.png          — SP+LG keypoint matches overlaid side-by-side
  imu.json             — raw IMU row (same content as SAVE_IMU_ROWS)
  trace.json           — full step-by-step structured data (see below)
```

**trace.json fields:**
- `ekf_before` / `ekf_after`: lat, lon, yaw, altitude, vel_n/e/d before and after the update
- `pf_state`: n_eff, spread_m (particle filter health)
- `pf_center`, `search_radius_m`: where the PF guided the search
- `first_pass_tiles`: ranked list of all tiles tested, with match count per tile
- `second_pass_tiles`: 8-neighbour expansion tiles (temporal only)
- `meta_tile_info`: which tiles went into the meta-tile, verification match count, verified flag
- `homography`: cs_shape, inliers, raw and look-ahead-corrected position
- `semantic.conf`: semantic confirmation confidence
- `gate_pass`, `method`, `r_used_sqrt`: gate decision and EKF noise used

Used by `pipeline_trace.ipynb`.

---

## Analysis notebooks and their requirements

| Notebook | Cells | Data source | Flags needed |
|----------|-------|------------|--------------|
| `live_analysis.ipynb` | 1–12, 15 | `results.csv` | None — always saved |
| `live_analysis.ipynb` | 13 | `analysis_extras.csv` | `SAVE_ANALYSIS_DATA=True` |
| `live_analysis.ipynb` | 14 | `timing_data.csv` | `SAVE_TIMING_DATA=True` |
| `pipeline_trace.ipynb` | all | `pipeline_data/frame_NNNN/` | `SAVE_PIPELINE_TRACE=True` |

---

## Combinations

### All flags are independent — any subset works

There are no hard conflicts. You can mix and match freely.

### Overlapping content

| Scenario | What happens |
|----------|-------------|
| `SAVE_PIPELINE_TRACE=True` + `SAVE_QUERY_FRAMES=True` | Query frame saved **twice**: once to `pipeline_data/frame_NNNN/query.jpg`, once to `flight_data/frame_NNNN.jpg`. No data loss, minor redundancy. |
| `SAVE_PIPELINE_TRACE=True` + `SAVE_IMU_ROWS=True` | IMU JSON saved twice to different paths. Same behaviour. |
| `SAVE_PIPELINE_TRACE=True` + `DEBUG_SAVE_METATILES=True` | Meta-tile saved twice: once to `pipeline_data/frame_NNNN/reference_tile.png` (per-run), once to global `outputs/metatiles/` with timestamp name. |

### Use with caution

| Scenario | Risk |
|----------|------|
| `ACCUMULATE_HISTORY=True` on a long run | Unbounded RAM growth — `searcher.history` stores every result dict including numpy arrays. Enable only for short notebook runs. |
| `SAVE_PIPELINE_TRACE=True` on a 970-frame run | ~150 MB disk + ~140 s overhead. Designed for short test runs (20–50 frames) or selected-frame reruns. |
| `DEBUG_SAVE_METATILES=True` in production | Writes to the global `outputs/metatiles/` folder, not the run directory. Files accumulate across runs and are not cleaned up automatically. |

### Recommended configurations

| Goal | Recommended flags |
|------|------------------|
| Standard run, minimal overhead | all `False` |
| PX4 GPS output for autopilot integration | `SAVE_ANALYSIS_DATA=True` |
| Full offline analysis (Cell 13 + 14) | `SAVE_ANALYSIS_DATA=True` + `SAVE_TIMING_DATA=True` |
| Research figure generation | `SAVE_PIPELINE_TRACE=True` (short run, 20–50 frames) |
| Raw data archive (re-process later) | `SAVE_QUERY_FRAMES=True` + `SAVE_IMU_ROWS=True` |
| Complete data capture | all `True` (aware of ~200 ms/frame overhead) |
| Interactive notebook session | `ACCUMULATE_HISTORY=True` + `DEBUG_SAVE_METATILES=True` |
