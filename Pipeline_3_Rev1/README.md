# Pipeline 3 — GPS-Free Drone Localization

Visual-inertial localization against a pre-built reference tilemap.  
No GPS required at inference time.

**Phase C results (50 frames, MSFS 2020 flight over Vejle, Denmark):**
- Online EKF mean error: **9.7 m**, median: **2.0 m**
- 49/50 frames (98%) under 50 m, 50/50 (100%) under 100 m
- Gate passes: 41/50 (82%)
- Speed: ~2.8 s/frame (RTX 5050, 1280 px cap, 2048 keypoints)

---

## Architecture

```
IMU CSV / SimConnect stream
        │
        ▼
Error-State EKF (10D)  ─────────── dead-reckoned lat/lon/heading/velocity
  [ekf_ins.py]                      position-error states updated every frame
  WMM2025 magnetic declination      computed from initial GPS at bootstrap
  [wmm_declination.py]
        │
        ▼
TemporalSearcher
  [temporal_searcher.py]
        │
   ┌────┴────┐
   │         │
Frame 0    Frame N (N ≥ 1)
   │         │
BestFirst  Particle filter predict
Search      → search region (radius)
   │         │
   └────┬────┘
        │
        ▼
SemanticTileScorer pre-filter
  → keep top-10 candidates before feature matching
        │
        ▼
MetaTileBuilder  [two-pass search]
  Pass 1: SuperPoint+LightGlue vs all tiles in radius
  Pass 2: 8-neighbour expansion around top-1
  → stitch top-K tiles into one meta-tile
        │
        ▼
VisualMeasurement  [visual_measurement.py]
  → rotate query to heading, dual homography (MAGSAC + DLT)
  → pick best of multiple pose methods
        │
        ▼
Quality gate  (CShape > 0.3, inliers > 20)
  PASS → camera look-ahead correction (−110 m along heading)
          → EKF.update_position(corrected_lat, corrected_lon, adaptive_R)
  FAIL → EKF coasts on IMU prediction
        │
        ▼
PositionEstimator  [position_estimator.py]
  → homography → pixel offset → GPS via tile geo-referencing
        │
        ▼
Particle filter update + resample
(guides search region only — EKF is the position estimate)
```

---

## Project Structure

```
Pipeline_3_Rev1/
├── config/
│   └── config.py               All paths, numeric constants, flags
│
├── src/                        Core library (import as `from src.X import Y`)
│   ├── ekf_ins.py              10D Error-State EKF; step_ekf(); preprocess_imu_csv()
│   ├── wmm_declination.py      WMM2025 magnetic declination & inclination
│   ├── temporal_searcher.py    Top-level frame processor; orchestrates everything
│   ├── best_first_search.py    Frame-0 cold-start exhaustive tile search
│   ├── meta_tile_builder.py    Two-pass tile search + meta-tile stitching
│   ├── particle_filter.py      Particle filter for search-region guidance
│   ├── geometric_matcher.py    SuperPoint+LightGlue wrapper
│   ├── visual_measurement.py   Heading rotation, dual homography, pose methods
│   ├── position_estimator.py   Homography → GPS via tile geo-referencing
│   ├── semantic_model.py       UNet++ EfficientNet-B3 segmentation model loader
│   ├── semantic_tile_scorer.py Histogram-based tile pre-filter
│   ├── semantic_confirmer.py   Centroid-based semantic double-confirmation
│   ├── tile_utils.py           TMS tile math, TileLoader, haversine distance
│   └── image_utils.py          load_image(), preprocess_query_frame() (semantic only)
│
├── runtime/
│   ├── run_pipeline.py         CLI entry point (file mode + SimConnect mode)
│   └── simconnect_adapter.py   Live MSFS 2020 data source via Python SimConnect
│
├── analysis/
│   ├── evaluate_run.py         Compute accuracy metrics from a run directory
│   ├── plot_trajectory.py      Map-view trajectory plot (GT vs estimated)
│   └── plot_diagnostics.py     Multi-panel quality/timing diagnostics plot
│
├── notebooks/
│   ├── test_temporal_pipeline.ipynb   Interactive tuning notebook (10 cells)
│   ├── live_analysis.ipynb            Standalone analysis for any run directory
│   └── diagnostics.ipynb             Deep-dive diagnostic plots (8 cells)
│
├── tests/
│   ├── test_10d_ekf.py         EKF unit tests (all pass)
│   ├── test_units.py           Unit-conversion and source-alignment tests
│   ├── test_particle_filter.py Particle filter tests
│   ├── test_meta_tile_builder.py
│   ├── test_semantic_confirmer.py
│   └── test_temporal_searcher.py
│
├── docs/
│   └── PIPELINE_07_04_2026.md     Full Phase C architecture documentation
│
└── outputs/                    All generated files (gitignored)
    ├── runs/                   One sub-directory per run_pipeline.py run
    ├── analysis/               Saved PNGs from live_analysis.ipynb
    ├── metatiles/              Debug meta-tile PNGs (DEBUG_SAVE_METATILES=True)
    └── ...
```

---

## Requirements & Setup

### Python environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install lightglue            # SuperPoint + LightGlue
pip install segmentation-models-pytorch timm
pip install pandas numpy scipy matplotlib opencv-python-headless
pip install pyproj h5py
# Optional (SimConnect live mode only):
pip install SimConnect mss pywin32
```

The reference venv is `.final_Pipeline_venv` at the workspace root and has all
of the above already installed.

### Data layout

All data lives **outside** `Pipeline_3_Rev1/`, typically one level up in
`All_In_One_Pipeline/`. By default the pipeline expects:

```
All_In_One_Pipeline/
├── REFERENCE_MAP_VEJLE_20260321_162024/
│   ├── aerial/16/{x}/{y}.png         512×512 orthophoto tiles (TMS zoom 16)
│   └── prediction/16/{x}/{y}.png     Semantic prediction tiles
├── Logs_Run_20260321_162024/
│   ├── imu_gps_log_20260321_162024.csv    IMU + GPS log (970 rows, 100 Hz)
│   └── images_20260321_162024/frame_*.jpg Query frames (1920×1079)
├── SemanticTerrainSegmentationModel/
│   └── best.pth                      UNet++ EfficientNet-B3 weights
├── Dataset_Preprocessing/
│   └── reference_features.h5         Pre-computed SuperPoint reference features
│                                     (3960 tiles × 2048 keypoints)
└── WMM2025COF/WMM2025COF/
    └── WMM2025.COF                   WMM2025 Gauss coefficients (loaded at runtime)
```

To point the pipeline at a different root directory, set the environment variable:

```bash
set PIPELINE3_DATA_ROOT=C:\path\to\All_In_One_Pipeline   # Windows
export PIPELINE3_DATA_ROOT=/path/to/All_In_One_Pipeline  # Linux/Mac
```

If the variable is not set, the parent of `Pipeline_3_Rev1/` is used automatically.

---

## Configuration

All tunable parameters live in `config/config.py`. Key values:

| Constant | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `"cuda"` | PyTorch device |
| `MAX_NUM_KEYPOINTS` | `2048` | SuperPoint keypoints per image |
| `MAX_ROTATED_DIMENSION` | `1280` | Long-edge cap after heading rotation (speed) |
| `IMU_SEARCH_RADIUS_METERS` | `500` | Frame-0 tile search radius |
| `FIRST_PASS_SEARCH_RADIUS_M` | `500` | Per-frame first-pass tile search radius |
| `QUALITY_GATE_CSHAPE` | `0.3` | Min CShape to accept visual update |
| `QUALITY_GATE_INLIERS` | `20` | Min inlier count to accept visual update |
| `VISUAL_POSITION_NOISE_M` | `50` | EKF measurement noise std (σ, metres) |
| `POSITION_PROCESS_NOISE_M` | `5` | EKF position process noise std per √s |
| `INITIAL_POSITION_VARIANCE_M` | `200` | Initial EKF position uncertainty (σ) |
| `SEMANTIC_PREFILTER_ENABLED` | `True` | Pre-filter tiles by semantic histogram |
| `SEMANTIC_PREFILTER_TOP_K` | `10` | Keep this many candidates before SP+LG |
| `METATILE_TOP_K` | `3` | Tiles to stitch into meta-tile |
| `DEBUG_SAVE_METATILES` | `False` | Save meta-tile PNGs (notebook/debug only) |
| `ACCUMULATE_HISTORY` | `False` | Accumulate full result history (notebook only) |
| `DIVERGENCE_POSITION_THRESHOLD_M` | `500` | Particle-filter divergence reset threshold |

### Adaptive measurement noise (hard-coded in `run_pipeline.py`)

| Condition | R (m²) | Std dev |
|-----------|--------|---------|
| Cold-start frame (first visual match after EKF bootstrap) | 10 000 | 100 m |
| High quality: CShape > 0.5 AND inliers > 100 | 900 | 30 m |
| Normal quality (gate passes, below high threshold) | 3 600 | 60 m |
| Bank > 20° (turn): multiply current R by | × 2.0 | — |
| Meta-tile not verified: multiply current R by | × 2.0 | — |
| Semantic confidence factor: R × max(0.5, 2.0 − 1.5 × sem_conf) | — | — |

The camera look-ahead correction constant `LOOKAHEAD_M = 110.0` is defined
directly in `run_pipeline.py` (and in notebook Cell 2) because it is dataset-specific.

---

## Magnetic Declination — WMM2025

At EKF bootstrap, `run_pipeline.py` calls `src/wmm_declination.py` to compute
the WMM2025 magnetic declination and inclination from the initial GPS coordinate:

```python
mag_dec_deg, mag_inc_deg = get_mag_field(lat0, lon0, alt_m=alt0)
ekf = ErrorStateEKF(..., mag_dec_deg=mag_dec_deg, mag_inc_deg=mag_inc_deg)
```

This replaces the hardcoded `mag_dec = 4°` used in earlier versions.
For Vejle, Denmark (55.7°N, 9.5°E, ~540 m, 2026.3) the computed values are
approximately `dec ≈ 4.2°`, `inc ≈ 70.5°`.

The WMM2025 coefficients are loaded from `WMM2025COF/WMM2025COF/WMM2025.COF`
(lazy-loaded and cached on first call, ~100 µs per call thereafter).

To verify the implementation:
```bash
python -c "from src.wmm_declination import get_mag_field; print(get_mag_field(43, 93, 65000, 2025.0))"
# Expected: (~0.50, ~64.10)  (matches WMM2025 test values)
```

---

## Three Ways to Use the Pipeline

### Part 1 — Interactive Notebook (tuning & research)

Open `notebooks/test_temporal_pipeline.ipynb` and run cells in order:

| Cell | Purpose |
|------|---------|
| 1 | Title / imports marker |
| 2 | Setup: imports, config overrides, `LOOKAHEAD_M`, adaptive-R thresholds |
| 3 | Load IMU CSV, run EKF warmup to `START_ROW`, create `live_ekf` instance |
| 4 | EKF sanity check (batch dead-reckoning only, no visual) |
| 5 | **Main closed-loop run** — 50 frames, prints per-frame results |
| 6 | Method distribution table |
| 7 | Trajectory map (GT vs estimated) |
| 8 | Error distribution histogram |
| 9 | Temporal error plot with bank-angle shading |
| 10 | Error CDF + quality gate analysis |

Variables populated after Cell 5 (`df_results`, `results`, `aligned`, etc.) are
required by `notebooks/diagnostics.ipynb` for deeper analysis.

**When to use**: Parameter tuning, algorithm debugging, result visualization.

---

### Part 2 — File Mode (batch processing from recorded data)

```bash
cd Pipeline_3_Rev1
python runtime/run_pipeline.py \
    --source file \
    --start-row 430 \
    --max-frames 300 \
    --run-id my_run_01
```

All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `file` | Data source: `file` or `simconnect` |
| `--imu-csv` | config path | Override IMU CSV path |
| `--frames-dir` | config path | Override query frames directory |
| `--run-id` | timestamp | Name for the output directory |
| `--output-dir` | `outputs/runs/` | Root directory for run outputs |
| `--max-frames` | all | Stop after N frames |
| `--start-row` | `0` | IMU CSV row to start from |
| `--debug` | off | Save meta-tile PNGs + full history |

Results are written to `outputs/runs/<run-id>/results.csv`.

**When to use**: Repeatable benchmarks, full-flight processing, CI.

---

### Part 3 — Live SimConnect Mode (MSFS 2020 real-time)

```bash
# Prerequisites:
#   - MSFS 2020 running with SimConnect enabled
#   - Python SimConnect + mss + pywin32 installed
#   - Drone flying over the reference map area

python runtime/run_pipeline.py --source simconnect --run-id live_001
```

`simconnect_adapter.py` runs a background thread that polls MSFS at ~10 Hz,
converts units (ft/s² → m/s², knots → m/s, MSFS left-handed body → standard NED),
and exposes the latest IMU row + captured screen frame to the main loop via
non-blocking accessors. Frame capture is time-gated at 5 fps.

The frame capture timestamp (`perf_counter`) is stored when each frame is grabbed
and returned as the third element of `get_latest_frame()` → used to compute
`inference_ms` (time from capture to GPS estimate) in `results.csv`.

At bootstrap the pipeline:
1. Waits for the first valid SimConnect GPS sample
2. Computes WMM2025 magnetic declination from the initial GPS coordinate
3. Initialises the EKF with the correct dec/inc for the flight area
4. Prints: `[run_pipeline] WMM2025 dec=X.XX°  inc=XX.XX°`

**When to use**: Real-time localization during live MSFS 2020 flight.

---

### Part 4 — Analyze a Run

Open `notebooks/live_analysis.ipynb`, set `RUN_DIR` in Cell 1, then
**Kernel → Restart and Run All**. It is fully self-contained.

| Cell | Content |
|------|---------|
| 1 | Setup & load `results.csv` |
| 2 | Summary metrics table (mean/median/min/max error, % under thresholds) |
| 3 | Error over time |
| 4 | Trajectory map (EKF vs GPS GT) |
| 5 | Gate health (CShape + inliers over time) |
| 6 | EKF position sigma convergence |
| 7 | Error CDF |
| 8 | Timing & performance (search time + `inference_ms` if SimConnect run) |

Works with both file-mode and SimConnect-mode `results.csv`. The `inference_ms`
column is `None` in file mode and displays automatically in SimConnect runs.

Alternatively use the CLI analysis scripts:

```bash
# Print accuracy table
python analysis/evaluate_run.py --run-dir outputs/runs/my_run_01

# Save trajectory map PNG
python analysis/plot_trajectory.py --run-dir outputs/runs/my_run_01

# Save multi-panel diagnostics PNG
python analysis/plot_diagnostics.py --run-dir outputs/runs/my_run_01
```

---

## Typical Workflow

```
1. Configure
   └── Edit config/config.py if data paths differ from defaults
       Set PIPELINE3_DATA_ROOT env var if needed

2. Explore & tune in the notebook
   └── notebooks/test_temporal_pipeline.ipynb
       Adjust LOOKAHEAD_M, R_HIGH, R_MED, QUALITY_GATE_* in Cell 2
       Run Cell 5 to see per-frame results
       Run diagnostics.ipynb for deeper analysis

3. Confirm with file mode
   └── python runtime/run_pipeline.py --source file --start-row 430 --max-frames 300

4. Analyze the run
   └── Open notebooks/live_analysis.ipynb, set RUN_DIR, Restart & Run All
       Or: python analysis/evaluate_run.py --run-dir outputs/runs/my_run_01

5. Live MSFS run
   └── python runtime/run_pipeline.py --source simconnect --run-id live_001
       Open notebooks/live_analysis.ipynb, set RUN_DIR to live_001, Run All
```

---

## Output Format

Each run writes `outputs/runs/<run-id>/results.csv` with one row per frame (31 columns):

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | int | Frame index (0-based within the run) |
| `timestamp` | float | Timestamp (seconds) |
| `image_name` | str | Frame filename stem (file mode) or `live_<id>` (SimConnect) |
| `final_lat` | float | EKF estimated latitude (°) after visual update |
| `final_lon` | float | EKF estimated longitude (°) after visual update |
| `heading_deg` | float | EKF heading (° true north) |
| `altitude_m` | float | EKF altitude (m) from barometer / pressure_altitude |
| `roll_deg` | float | EKF roll (°) |
| `pitch_deg` | float | EKF pitch (°) |
| `vel_n` | float | EKF north velocity (m/s) |
| `vel_e` | float | EKF east velocity (m/s) |
| `vel_d` | float | EKF down velocity (m/s) |
| `gps_lat` | float | GPS ground-truth latitude (°); NaN if unavailable |
| `gps_lon` | float | GPS ground-truth longitude (°); NaN if unavailable |
| `gps_alt_m` | float | GPS ground-truth altitude (m) |
| `method` | str | Visual localization method: `cold_start`, `temporal`, `imu_fallback` |
| `gate_pass` | int | 1 if quality gate passed and EKF was updated, 0 otherwise |
| `search_time_s` | float | Wall-clock time for this frame's visual search (s) |
| `cs_shape` | float | CShape score (visual match quality, 0–1) |
| `inliers` | int | RANSAC inlier count |
| `semantic_conf` | float | Semantic histogram match confidence (0–1) |
| `homo_lat` | float | Raw homography-derived latitude (before look-ahead correction) |
| `homo_lon` | float | Raw homography-derived longitude (before look-ahead correction) |
| `homo_corrected_lat` | float | Look-ahead-corrected latitude fed to EKF |
| `homo_corrected_lon` | float | Look-ahead-corrected longitude fed to EKF |
| `meta_tile_verified` | int | 1 if meta-tile second-pass verification passed |
| `ekf_pos_sigma` | float | EKF position uncertainty σ = √max(P[8,8], P[9,9]) (m) |
| `r_used_sqrt` | float | √R used for EKF position update (m); None if no update |
| `tiles_tested` | int | Number of tiles evaluated by feature matcher |
| `verification_matches` | int | Inlier count from meta-tile second-pass verification |
| `inference_ms` | float | End-to-end latency from frame capture to GPS estimate (ms); None in file mode |

---

## Unit Tests

```bash
cd Pipeline_3_Rev1
python -m pytest tests/ -v
```

| Test file | What it covers | Status |
|-----------|---------------|--------|
| `test_10d_ekf.py` | ErrorStateEKF predict/update, 10D state, `update_position()` | All pass |
| `test_units.py` | Accel unit conversion, airspeed kts→m/s, `FileSource` alignment | All pass |
| `test_particle_filter.py` | Particle predict, update, resample, TMS math | All pass |
| `test_meta_tile_builder.py` | Two-pass search, meta-tile dict structure | Pre-existing mock issues |
| `test_semantic_confirmer.py` | Centroid extraction/matching | Pre-existing API mismatch |
| `test_temporal_searcher.py` | Frame-0/frame-N dispatch, trajectory saving | Pre-existing mock issues |

---

## Performance & Known Limitations

- **Speed**: ~2.8 s/frame on RTX 5050 (`MAX_ROTATED_DIMENSION=1280`,
  `MAX_NUM_KEYPOINTS=2048`). Cut to ~1.5 s by caching query SuperPoint
  features across tile candidates within a frame (not yet implemented).

- **Camera look-ahead (`LOOKAHEAD_M = 110`)**: Empirically tuned for this
  specific MSFS flight. The MSFS camera has a fixed forward tilt so the image
  center corresponds to ground ~110 m ahead of the drone. A different drone,
  camera mount angle, or altitude will require re-calibration using the
  Lookahead Calibration cell in `diagnostics.ipynb`.

- **Cold-start EKF trust (`R_COLD_START = 10 000 m²`)**: The first visual match
  after bootstrap uses a 100 m std dev rather than the default 30 m. This reduces
  over-trust in the first (cold-start) measurement while still converging quickly.
  Frame 1 onward uses the normal adaptive noise schedule.

- **Domain mismatch**: Query frames are 3D rendered perspective images;
  reference tiles are orthophotos. SP+LG quality collapses in areas with
  large buildings, heavy tree canopy, or when the drone banks sharply (frames
  39–47 in the reference run show this).

- **Reference map coverage**: The tilemap covers only the Vejle area
  (TMS zoom 16, X: 34482–34547, Y: 45003–45062). Frames outside this
  bounding box fall back to IMU dead-reckoning. Use `START_ROW` to skip
  frames before the drone enters the mapped area (auto-detected in the
  notebook Cell 3).

- **HDF5 feature store** (`reference_features.h5`): Pre-computed with
  `MAX_NUM_KEYPOINTS=2048`. If you change this constant in config, the store
  must be rebuilt by re-running the `Dataset_Preprocessing` notebook.

- **WMM2025 validity**: The WMM2025 model is valid through 2029.9. The
  `WMM2025.COF` coefficient file must be present at
  `All_In_One_Pipeline/WMM2025COF/WMM2025COF/WMM2025.COF`.
