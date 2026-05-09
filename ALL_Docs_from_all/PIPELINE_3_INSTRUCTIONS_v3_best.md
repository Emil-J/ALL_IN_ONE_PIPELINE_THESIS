# PIPELINE 3: BEST-FIRST WITH TEMPORAL PARTICLE TRACKING
## Audited Implementation Instructions for Claude / VS Code Agent

---

## MISSION OVERVIEW

You are extending Pipeline 1 (Best-First Search) with temporal particle filtering
for video stream localization. This provides 10× speedup on subsequent frames
while maintaining smooth, consistent trajectories.

**Core Strategy**:
- **Frame 0**: Use Pipeline 1 best-first search (cold start)
- **Frame 1+**: Propagate particles with IMU motion, use particle cloud to guide
  focused search, update particles with match results

**Key Innovation**: Temporal consistency + multi-hypothesis tracking + adaptive
search region + two-pass 8-neighbour meta-tile search + semantic
double-confirmation.

**Implementation Approach**: Build on top of completed Pipeline 1. Add modules
incrementally. Test thoroughly at each step.

---

## EXTENDED LOCALIZATION ARCHITECTURE (PIPELINE 3 VISION)

> This section describes the full localization loop that Pipeline 3 targets.
> It augments the particle filter + temporal tracking core with a two-pass
> tile search, meta-tile construction, and semantic double-confirmation.
> Implementation is described sequentially but designed for parallel
> execution in production.

### Parallel entry on each frame

On receiving a new query frame, two branches launch concurrently:

- **Branch A** — Semantic pre-check (low-confidence, informational)
- **Branch B** — Two-pass SuperPoint+LightGlue tile search (primary path)

The IMU position estimate (from EKF dead reckoning) defines the search region
for both branches. Branch A result is stored and consumed in Stage 4.
Branch B drives all primary decisions.

---

### Branch A — Semantic pre-check

#### A1. Semantic segmentation of query frame

Run the query frame through UNet++ + EfficientNet-B3 + scSE.

Output: semantic map (class mask, 512×512 px after padding)

Classes (indices 0–5):
- 0: waterbodies — rgb(4, 4, 255)
- 1: forest_trees — rgb(0, 167, 2)
- 2: land — rgb(243, 255, 150)
- 3: railway — rgb(193, 105, 53)
- 4: roads — rgb(255, 0, 231)
- 5: buildings — rgb(150, 150, 150)

#### A2. Centroid-based feature matching

Using the MDPI centroid-based approach (doi:10.3390/rs17101671):

1. Extract semantic centroids from the query semantic map
2. Retrieve all reference tile predictions within the IMU uncertainty
   radius (~300 m)
3. Attempt centroid matching between query centroids and reference centroids
4. Compute confidence score

**Expected outcome**: Low confidence due to domain gap between 3D MSFS
perspective imagery and 2D nadir reference tiles. Result is logged and
stored for use in Stage 4 semantic double-confirmation. The pipeline does
not branch on this score — it is informational only.

---

### Branch B — Two-pass SuperPoint+LightGlue tile search

#### B1. First-pass tile search

1. Define search region: all tiles within ~300 m radius of IMU estimate
2. For each candidate tile, run SuperPoint+LightGlue against the query frame
3. Record keypoint match count per tile
4. Rank candidates descending by match count
5. Select **top-1** candidate (primary pivot), optionally top-2

#### B2. Second-pass 8-neighbour search

Using top-1 from B1 as pivot:

1. Retrieve the **8 grid neighbours** of top-1 tile
   (N, NE, E, SE, S, SW, W, NW) plus top-1 itself = **9 tiles total**
2. Run SuperPoint+LightGlue on each of the 9 tiles individually
   against the query frame
3. Rank all 9 by match count
4. Select **top-3** tiles from this ranked list

> If top-2 was also retained from B1, optionally extend to the union of
> 8-neighbourhoods of both top-1 and top-2 before ranking.

#### B3. Meta-tile construction

Combine top-3 tiles from B2 into a single composite reference image:

- Arrange tiles spatially according to their grid positions (x, y)
- Irregular shapes are acceptable — tiles need not form a perfect rectangle
- Pad empty regions with **black (RGB 0, 0, 0)**
- The result is the **meta-tile**

#### B4. Meta-tile persistence

Save the meta-tile to disk immediately after construction:

```
outputs/metatiles/metatile_{query_timestamp:.3f}.png
```

Filename timestamp corresponds to the **capture timestamp of the query
frame** (relative flight seconds, matching frame naming convention
`frame_0.523.jpg` → `metatile_0.523.png`). Saving occurs before
verification — always persist regardless of downstream outcome.

---

### Stage 3 — Meta-tile verification

Run SuperPoint+LightGlue between the **query frame** and the **meta-tile**:

1. Extract keypoints and descriptors from both
2. Match descriptors
3. Count inlier matches (after ratio test / RANSAC)

**Decision**:
- If match count ≥ `METATILE_MATCH_THRESHOLD` → proceed to Stage 4
- If match count < threshold → fall back to IMU estimate (Stage 5b)

> Threshold to be tuned empirically. Starting point: 25 inlier matches
> for a 512×288 query against a 2–3 tile meta-tile. Expect lower counts
> than aerial-to-aerial matching due to MSFS domain gap.

---

### Stage 4 — Semantic double-confirmation

With a verified meta-tile:

1. Run meta-tile through semantic model → meta-tile semantic map
2. Extract semantic centroids from meta-tile semantic map
3. Run centroid-based matching (MDPI method) between:
   - Query semantic map (stored from Branch A — do not re-run inference)
   - Meta-tile semantic map
4. Semantic shapes should agree closely if meta-tile is correctly localised

This provides a second independent confidence signal before committing to
a pose estimate. It also cross-validates the feature matching result against
the semantic understanding of the scene.

---

### Stage 5 — Pose estimation → GPS coordinate

Using the verified meta-tile and inlier correspondences from Stage 3:

1. Estimate homography H between query frame and meta-tile
   (planar Homography + RANSAC — recommended starting point for nadir
   footage over flat terrain)
2. Use H and known tile geospatial coordinates to estimate drone
   position (lat/lon) and heading
3. Output estimated GPS coordinate

> GPS output is used **for benchmarking only**. It is never fed back
> as a navigation input. GPS is used solely for EKF Frame 0
> initialisation and for evaluation against ground truth.

---

### Stage 5b — IMU fallback (if Stage 3 fails)

If meta-tile verification fails:

- Log failure event: timestamp, match count, IMU estimate, particle spread
- Return IMU dead-reckoning position as current best estimate
- Flag position confidence as LOW
- On next frame, expand search radius by configurable step (e.g. +100 m)
- Particle filter continues propagating — do not reinitialise unless
  `check_divergence()` fires

---

### File outputs summary

| Path | Description |
|------|-------------|
| `outputs/metatiles/metatile_{ts:.3f}.png` | Meta-tile composite, saved after B3 |
| `outputs/logs/pipeline3_run_{ts}.jsonl` | Per-frame log: IMU, match counts, decision, position |
| `outputs/semantic/query_{ts:.3f}.png` | Query semantic map (optional, debug) |
| `outputs/trajectories/trajectory.csv` | Full trajectory from `save_trajectory()` |
| `outputs/trajectories/summary.json` | Performance summary |

---

### Notes on parallelism

The sequential description above is for clarity. In production:

- Branch A and Branch B launch concurrently on query frame arrival
- Branch A result (query semantic map + centroid score) is stored and
  retrieved in Stage 4 — no second inference pass on the query frame
- Branch B drives the main decision path and the particle filter update
- Pose estimation (Stage 5) can run asynchronously, writing the GPS
  coordinate to the EKF measurement queue after completion

---

## WHAT THIS VERSION FIXES

This version is stricter than the previous two documents. It keeps the strong parts
of the audited file, but removes ambiguity that can cause an agent to wander, overbuild,
or silently break the pipeline.

### Improvements over previous versions

1. **Agent-agnostic wording**
   - This file is written for Claude in VS Code, but it does not hard-bind behaviour
     to a specific model version string.

2. **Hard execution boundaries**
   - Do **not** replace SuperPoint+LightGlue with a different matcher unless explicitly requested
   - Do **not** replace the semantic model unless explicitly requested
   - Do **not** rewrite Pipeline 1 from scratch if working code already exists
   - Do **not** use GPS after Frame 0 except for benchmarking/evaluation
   - Do **not** treat semantic confirmation as a hard gate until empirical calibration is logged

3. **Clear deliverables**
   - Every module must expose a minimal stable API
   - Every stage must log enough data to debug failures offline
   - Every frame result must be serializable to JSONL

4. **Safer fallback logic**
   - If meta-tile verification fails, keep tracking alive, widen search gradually,
     and avoid full reset unless divergence criteria are actually met

5. **More realistic implementation priority**
   - First make the pipeline correct and debuggable
   - Then make it faster
   - Then make semantic confirmation stricter if it proves useful

---

## NON-NEGOTIABLE CONSTRAINTS

- **Primary objective**: improve localization robustness relative to IMU-only drift while keeping runtime compatible with the current ~2.18 fps query stream
- **Primary visual matcher**: SuperPoint + LightGlue
- **Semantic model**: existing UNet++ EfficientNet-B3 scSE checkpoint
- **Reference map source**: existing Vejle TMS tiles and prediction masks
- **Navigation rule**: GPS is not a navigation input beyond Frame 0 initialization
- **Search policy**: first use focused local search; only expand when evidence is weak
- **Engineering rule**: prefer measurable, logged, testable behaviour over clever but opaque logic

---

## REQUIRED DELIVERABLES

At the end of the implementation, the workspace must contain:

1. Working Python modules for particle filtering, temporal search, meta-tile construction, and semantic confirmation
2. Unit tests for each new module
3. One end-to-end notebook for sequence evaluation
4. JSONL per-frame logs
5. Saved meta-tiles with timestamp-aligned filenames
6. CSV trajectory export
7. Summary metrics comparing:
   - IMU-only baseline
   - Pipeline 1 cold-start performance
   - Pipeline 3 temporal performance

---

## DEFINITION OF DONE

The implementation is only considered done when all of the following are true:

- Frame 0 uses the Pipeline 1 search path and returns a valid localization result
- Frame 1+ uses particle-guided focused search
- Meta-tile files are written for every processed frame after construction
- Verification success/failure is logged per frame
- Semantic confirmation is logged per frame even if used only as a soft confidence signal
- Failure cases do not crash the sequence run
- The notebook can process a multi-frame sequence and export summary metrics
- The code clearly separates **cold start**, **temporal tracking**, **verification failure**, and **divergence reset** states

---

## IMPLEMENTATION PHASES

### Phase A — Correctness first

Implement the modules with clear APIs and deterministic outputs. Use conservative defaults.
Do not optimize prematurely.

### Phase B — Sequence stability

Run on a short real sequence. Inspect:
- verification match counts
- particle spread
- fallback frequency
- error timeline
- meta-tile geometry correctness

### Phase C — Runtime optimization

Only after correctness is confirmed:
- reduce redundant inference
- cache loaded tiles
- parallelize safe branches
- reduce particle count only if accuracy does not regress materially

---

## MANDATORY LOGGING SCHEMA

Each processed frame must write one JSONL record with at least:

```json
{
  "timestamp": 0.523,
  "frame_name": "frame_0.523.jpg",
  "mode": "cold_start|temporal_tracking|imu_fallback|reset",
  "imu_lat": 55.7,
  "imu_lon": 9.5,
  "imu_heading": 180.0,
  "dt": 0.59,
  "first_pass_candidates": 12,
  "top1_tile": [34500, 45030],
  "top3_tiles": [[34500, 45030, 81], [34501, 45030, 64], [34500, 45031, 53]],
  "verification_matches": 27,
  "meta_tile_verified": true,
  "semantic_confidence": 0.41,
  "particle_position_std_m": 63.2,
  "particle_heading_std_deg": 11.8,
  "n_eff": 62.7,
  "estimated_lat": 55.7001,
  "estimated_lon": 9.5002,
  "estimated_heading": 178.6,
  "used_gps_feedback": false
}
```

If a field is unavailable for a frame, log `null` rather than omitting it.

---

## IMPORTANT CALIBRATION RULE

Because of the MSFS-to-aerial domain gap, treat the following as **initial hypotheses**,
not truths:

- `METATILE_MATCH_THRESHOLD = 25`
- semantic confirmation is useful as a strong confidence gate
- top-3 neighbourhood composition is always better than top-1

These must be validated on logged runs. Until then:
- semantic confirmation should remain **soft**
- fallback should prefer IMU continuity over aggressive reset
- thresholds should be tuned from collected histograms, not intuition

---

## HARD FAILURE MODES TO HANDLE

The implementation must explicitly handle these cases:

1. No first-pass tiles found in the search region
2. Top-1 tile exists but one or more 8-neighbours are missing at map edge
3. Meta-tile builds successfully but verification returns too few matches
4. Semantic segmentation runs but produces too few centroids for useful comparison
5. Particle filter collapses numerically or all weights become zero
6. Timestamp mismatch between CSV row and image filename rounding
7. Frame is missing even though the CSV row exists
8. Query frame shape is unexpected or corrupted

In all cases, fail gracefully, log the reason, and continue sequence processing where possible.

---

## PREREQUISITES

**Required**: Pipeline 1 must be fully implemented and working.

Verify Pipeline 1:
```python
from src.best_first_search import BestFirstSearcher
# If this imports successfully, Pipeline 1 is ready
```

---

## WORKSPACE CONTEXT

### Additional Files Needed from User

**USER: In addition to Pipeline 1 requirements, please provide:**

```
=== IMPLEMENTATION DECISION ===
**APPROACH**: Build Pipeline 1 from scratch, then extend it with temporal
particle tracking for Pipeline 3.
**LOCATION**: Implement in Pipeline_3_Rev1 folder (currently empty).

===Pipeline 1 Answers===

=== SEMANTIC MODEL INFORMATION ===
Model file path: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\SemanticTerrainSegmentationModel\best.pth
Model architecture: UNet++ with EfficientNet-B3 encoder + scSE attention modules
Number of classes: 6
Class names and indices (from legend.txt):
  0: waterbodies - rgb(4, 4, 255)
  1: forest_trees - rgb(0, 167, 2)
  2: land - rgb(243, 255, 150)
  3: railway - rgb(193, 105, 53)
  4: roads - rgb(255, 0, 231)
  5: buildings - rgb(150, 150, 150)
Input image size: 512x512 pixels
Output format: Semantic segmentation mask (H×W) with class indices (0-5)
How to load and run inference: Reconstruct UNet++ architecture in memory,
  load weights from best.pth
  - Training details in: SemanticTerrainSegmentationModel/Semantic_Model_QGIS_8_Class_Rev6.ipynb
  - Model config: SemanticTerrainSegmentationModel/config.json

=== IMU ESTIMATOR INFORMATION ===
**Location**: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\MSFS2020_IMU_Pipeline
**Main estimator**: ekf_ins.py (Error-State Extended Kalman Filter)
**Data logger**: data_logger.py (collects sensor data from MSFS via SimConnect)

IMU log file: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\imu_gps_log_20260321_162024.csv

CSV Column Classification (CRITICAL - read documentation in MSFS2020_IMU_Pipeline for full details):
**🟢 SENSOR INPUTS** (actual sensor measurements - OK to use in algorithms):
  - accel_x, accel_y, accel_z: IMU accelerometer (ft/s² → convert to m/s² via ×0.3048)
  - gyro_x, gyro_y, gyro_z: IMU gyroscope (rad/s, use directly)
  - pitch, bank: Attitude (radians, Python SimConnect auto-converts from DEGREES SimVars)
    NOTE: Used only for gravity synthesis (MSFS accel has no gravity component)
  - barometer_pressure: Barometric pressure (mbar → altitude via ISA formula)
  - heading_magnetic: Magnetic compass (radians, auto-converted from PLANE_HEADING_DEGREES_MAGNETIC)
  - airspeed_true: True airspeed (converted knots→m/s in data_logger.py before CSV storage)
  
**🔴 GROUND TRUTH** (simulator outputs - ONLY for evaluation and Frame 0 initialization):
  - latitude, longitude, altitude: GPS position (used ONCE for Frame 0, then only for evaluation)
  - heading: True heading (evaluation only)
  - ground_velocity: Ground speed (evaluation only)
  - vertical_speed: Climb rate (evaluation only)

**🟡 METADATA** (not used):
  - timestamp, ap_master, ap_alt_hold, ap_airspeed_hold, ap_nav_hold, throttle_pos

How to read EKF outputs (from ekf_ins.py output CSV):
  - Position estimate: latitude, longitude (degrees), altitude (meters)
  - Velocity estimate: vel_n, vel_e, vel_d (m/s in NED frame) ✅ YES, velocity IS output!
  - Heading estimate: yaw (degrees, derived from quaternion)
  - Position covariance: NOT currently output (internal to self.P matrix)
    - Could be extracted from self.P[0:3, 0:3] if modified
  - Heading uncertainty: NOT currently output
    - Could be extracted from orientation error covariance if modified

How to get gyro_z (yaw rate):
  - Option A: Use raw gyro_z from CSV (sensor input, rad/s)
  - Option B: Differentiate yaw output from EKF (degrees → rad/s)
  - Recommendation: Use raw gyro_z from CSV (it's a sensor value, not ground truth)

Streaming modification needed:
  - Current: Batch mode (processes entire CSV at once)
  - Required: Frame-by-frame streaming (process one timestep, return state immediately)
  - ✅ ALREADY COMPATIBLE: EKF processes sequentially, maintains state internally
  - Modification: Remove pandas dependency, call ekf.get_state() after each update
  - See ekf_ins.py lines 643-700 for main loop structure

Frame 0 initialization:
  - EKF uses FIRST GPS reading (latitude, longitude from CSV row 0) as reference point (lat0, lon0)
  - Altitude from barometer (NOT GPS altitude)
  - Heading from magnetometer (heading_magnetic)
  - Velocity from airspeed (if available) or zero
  - After Frame 0, GPS is NEVER used again (pure dead reckoning)
  - This explains why timestamp 0.021s has no image (first GPS is initialization only)

MSFS SimVars reference: https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Aircraft_SimVars/Aircraft_Misc_Variables.htm

=== FEATURE MATCHER INFORMATION ===
Using: **SuperPoint + LightGlue** (LightGlue is successor to SuperGlue)
Model weights: Auto-downloaded on first use (stored in torch cache ~/.cache/torch/hub/)
  - No manual weight files needed
  
How to initialize (from feature_matching_comparison_v2.ipynb):
```python
from lightglue import SuperPoint, LightGlue
from lightglue.utils import numpy_image_to_torch

class SuperPointLightGlueMatcher:
    def __init__(self, max_num_keypoints=2048, device='cuda'):
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)
    
    @torch.no_grad()
    def match(self, img1, img2):
        # Convert to grayscale tensors (B, 1, H, W)
        tensor1 = to_tensor_gray(img1, self.device)
        tensor2 = to_tensor_gray(img2, self.device)
        
        # Extract features
        feats0 = self.extractor.extract(tensor1)
        feats1 = self.extractor.extract(tensor2)
        
        # Match
        matches = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Parse output
        kpts1 = feats0['keypoints'][0].cpu().numpy()
        kpts2 = feats1['keypoints'][0].cpu().numpy()
        matches0 = matches['matches0'][0].cpu().numpy()
        scores = matches['matching_scores0'][0].cpu().numpy()
        
        # Build match pairs
        valid = matches0 >= 0
        match_pairs = np.column_stack([np.where(valid)[0], matches0[valid].astype(int)])
        
        return {
            'keypoints1': kpts1,
            'keypoints2': kpts2,
            'matches': match_pairs,
            'match_scores': scores[valid],
            'num_matches': len(match_pairs)
        }
```

Input format: Grayscale or RGB images (any size, will be processed internally)
Output format: Dict with keypoints1, keypoints2, matches (Nx2 array), match_scores
Installation: `pip install lightglue` (torch, torchvision already installed)
Reference implementation: feature_matching_comparison_v2.ipynb (cells with SuperPoint+LightGlue)

=== REFERENCE TILES INFORMATION ===
**Tiles directory**: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\REFERENCE_MAP_VEJLE

**Directory structure**:
REFERENCE_MAP_VEJLE/
├── aerial/          ← RGB aerial imagery
│   ├── 16/         ← Zoom level 16 tiles
│   │   ├── 34494/  ← X-coordinate directories
│   │   │   ├── 45025.png
│   │   │   ├── 45026.png
│   │   │   └── ...
│   │   ├── 34495/
│   │   └── ... (34494 to 34508)
│   └── metatiles/  ← 4×4 tile mosaics (30 files)
│       └── 16_&&_34493_20492_&&_34497_20496.png (example)
└── prediction/      ← Semantic segmentation masks (SAME structure as aerial/)
    ├── 16/
    └── metatiles/

**Tile naming pattern**: `{type}/16/{x}/{y}.png`
  - Example aerial: aerial/16/34500/45025.png
  - Example prediction: prediction/16/34500/45025.png
  
**Tile format**: .png (PNG images)
**Tile size**: 512×512 pixels
**Zoom level**: 16 (fixed)
**Tile coordinate ranges**:
  - X: 34494 to 34508 (15 tiles wide)
  - Y: 45025 to 45042 (18 tiles tall)
**Total tiles**: 270 tiles per type (aerial + prediction)
  - Calculation: 15 × 18 = 270
**Geographic coverage**: Vejle, Denmark (exact bounds in tile coordinates above)

**Metatile naming**: `16_&&_{x_start}_{y_start}_&&_{x_end}_{y_end}.png`
  - Each metatile: 4×4 individual tiles stitched together
  - Grid: 5 columns × 6 rows = 30 metatiles per type

**Metadata file** (optional): reference_tiles_metadata.csv
  - Contains: file_path, tile_x, tile_y, lat, lon, zoom (for each tile)

**Important notes**:
  - Each aerial tile has corresponding prediction mask at same coordinates
  - Predictions generated from semantic model (best.pth) during preprocessing
  - Stored in inference_results.json with class distribution stats

=== QUERY FRAMES INFORMATION ===
**Query frames directory**: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\images_20260321_162024

**Frame naming convention**: frame_{timestamp}.jpg
  - Example: frame_0.523.jpg, frame_1.113.jpg, frame_1.642.jpg
  - Timestamp has 3 decimal places (rounded from full precision in CSV)

**Timestamp matching strategy**:
  - CSV has full precision (e.g., 0.5229570865631104)
  - Frames have 3 decimals (e.g., 0.523)
  - **MATCHING METHOD**: Round CSV timestamp to 3 decimals for matching
  - Tolerance: Exact match after rounding (or ±0.001s if needed)

**Missing first frame**:
  - CSV row 0 (timestamp 0.021s) has NO corresponding image
  - Reason: First GPS reading used for EKF initialization only (lat0, lon0, heading0)
  - Processing starts from SECOND CSV row (timestamp ~0.523s, first actual frame)
  
**Image size**: 1920×1079 pixels (16:9 aspect ratio)
**Frame rate**: ~2.18 fps (calculated from timestamps)
**Total frames**: ~860 images (exact count: see directory listing)

**Image preprocessing** (for 512×512 semantic model input):
  - **METHOD**: Resize to 512×288, then pad top/bottom with black to 512×512
  - Steps:
    1. Resize 1920×1079 → 512×288 (preserves aspect ratio)
    2. Create 512×512 black canvas
    3. Center-paste 512×288 image (pad 112 pixels top, 112 pixels bottom)
  - This avoids cropping detail and prevents distortion

**Ground truth**: Same CSV file (imu_gps_log_20260321_162024.csv)
  - latitude, longitude columns have GPS coordinates for each frame
  - Use for evaluation ONLY (not for localization algorithm)

**CRITICAL**: Query frames have NO pre-computed prediction masks
  - REFERENCE_MAP_VEJLE has predictions/ folder
  - Query frames (Logs_Run_20260321_162024) do NOT have predictions
  - Must run semantic model inference on query frames during localization

===Pipeline 2 Additional Answers (Temporal Tracking)===

=== TEMPORAL DATA INFORMATION ===
Query video frame rate: ~2.18 fps (calculated from frame timestamps)
Frame timestamp format: Floating point seconds (e.g., 0.523, 1.113, 1.642)
Frame naming convention: Sequential by timestamp (see Pipeline 1 answers)
IMU data frequency: ~50 Hz in CSV (but only aligned with frames at ~2.18 Hz for processing)

How to read velocity from IMU/EKF:
  - EKF output CSV has: vel_n, vel_e, vel_d (m/s in NED frame)
  - Combine: velocity_mps = sqrt(vel_n² + vel_e²) for horizontal speed
  - Or use vel_n, vel_e separately for NE velocity components
  
How to read gyro_z (yaw rate):
  - Option A: Raw sensor from input CSV: gyro_z column (rad/s)
  - Option B: Differentiate yaw from EKF output: d(yaw)/dt
  - Recommendation: Use raw gyro_z from input CSV (it's a sensor measurement)
  - Convert to deg/s: gyro_z_rad * (180/π) = gyro_z_dps

Expected IMU update rate relative to frames:
  - Process IMU update when frame arrives (~2.18 Hz)
  - Between frames, EKF continues integrating internally (50 Hz sensor data)
  - For particle prediction, use dt = time since last frame (~0.46s average)

=== PARTICLE FILTER TUNING ===
Expected position drift per second: 0.66 m/s (from EKF performance analysis)
Expected heading drift per second: 0.025 deg/s (1.5 deg/min) (from EKF performance)

Acceptable position uncertainty (from EKF evaluation metrics):
  - Mean error: 165.78 m
  - Median error: 179.44 m
  - Std deviation: 78.73 m
  - **Particle filter recommendation**: Use ±100m as initial uncertainty, ±200m as max acceptable
  
Acceptable heading uncertainty (from EKF evaluation):
  - Mean error: 13.56°
  - Median error: 11.64°
  - Std deviation: 28.86°
  - **Particle filter recommendation**: Use ±15° as initial uncertainty, ±30° as max acceptable

===Additional Context===
Documentation: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\ALL_Docs_from_all
  - PIPELINE_TECHNICAL_DOCUMENTATION.md (complete IMU/EKF analysis)
  - LOCALIZATION_ROADMAP.md (project overview)
  - UNIT_ANALYSIS_AND_FIXES.md (coordinate system fixes)
  - notes from me.txt (chronological development notes)

Model training details: SemanticTerrainSegmentationModel/Semantic_Model_QGIS_8_Class_Rev6.ipynb
Feature matching comparison: feature_matching_comparison_v2.ipynb (DeDoDe vs SuperPoint+LightGlue)
Known issues: Domain shift between aerial reference tiles and MSFS query frames (see user memory)
Performance requirements: Real-time capable (~2fps processing) with 10× speedup after Frame 0
```

---

## PROJECT STRUCTURE ADDITIONS

Extend Pipeline 1 structure:

```
best_first_localization/
├── ... (all Pipeline 1 files)
├── src/
│   ├── ... (existing modules)
│   ├── particle_filter.py           # NEW: Particle filter implementation
│   ├── temporal_searcher.py         # NEW: Temporal-aware searcher (updated for MetaTileBuilder)
│   ├── meta_tile_builder.py         # NEW: Two-pass search + meta-tile construction
│   ├── semantic_confirmer.py        # NEW: Centroid-based double-confirmation
│   └── trajectory_smoother.py       # NEW: Post-processing smoother
├── notebooks/
│   ├── test_pipeline.ipynb          # (existing)
│   └── test_temporal_pipeline.ipynb # NEW: Temporal testing
├── tests/
│   ├── ... (existing tests)
│   ├── test_particle_filter.py      # NEW
│   ├── test_temporal_searcher.py    # NEW
│   └── test_meta_tile_builder.py    # NEW
└── outputs/
    ├── metatiles/                   # NEW: Timestamped meta-tile composites
    ├── semantic/                    # NEW: Query semantic maps (debug)
    ├── logs/                        # NEW: Per-frame JSONL logs
    └── trajectories/                # NEW: Trajectory outputs
```

---

## MODULE IMPLEMENTATION ORDER

Implement in this exact order, building on Pipeline 1:

---

## MODULE 8: src/particle_filter.py

**Purpose**: Particle filter for temporal state tracking.

**Implementation Requirements**:

### 8.1: Class Particle

**Data Structure**:
```python
@dataclass
class Particle:
    x: float           # Position in tile X coordinate
    y: float           # Position in tile Y coordinate  
    heading: float     # Heading in degrees [0, 360)
    weight: float      # Particle weight [0, 1]
```

**Methods**:
- `to_latlon(zoom)`: Convert (x, y) to (lat, lon)
- `from_latlon(lat, lon, zoom)`: Create from GPS

### 8.2: Class ParticleFilter

**Initialization**:
- `__init__(num_particles, initial_position, initial_heading, initial_spread, zoom)`
- initial_position: (lat, lon) or (tile_x, tile_y)
- initial_spread: dict with 'position_meters' and 'heading_degrees'
- Generate num_particles around initial with Gaussian spread
- All weights = 1/num_particles initially

**Configuration Parameters** (add to config.py):
```python
# Particle Filter Configuration
NUM_PARTICLES = 100
PROCESS_NOISE_POSITION_M = 5.0    # Position uncertainty per step
PROCESS_NOISE_HEADING_DEG = 2.0   # Heading uncertainty per step
MEASUREMENT_NOISE_POSITION_M = 50.0  # Expected match position error
MEASUREMENT_NOISE_HEADING_DEG = 10.0  # Expected match heading error
RESAMPLE_THRESHOLD = 0.5  # Resample when N_eff < threshold * num_particles
```

**Methods**:

### 8.3: predict(dt, velocity_mps, gyro_z_dps)
- Propagate each particle with IMU motion
- For each particle:
  ```
  # Kinematic update
  dx = velocity_mps * cos(particle.heading_rad) * dt
  dy = velocity_mps * sin(particle.heading_rad) * dt
  particle.x += dx / TILE_SIZE_METERS
  particle.y += dy / TILE_SIZE_METERS
  particle.heading += gyro_z_dps * dt
  particle.heading = particle.heading % 360  # Wrap to [0, 360)
  
  # Add process noise
  particle.x += random.normal(0, PROCESS_NOISE_POSITION_M / TILE_SIZE_METERS)
  particle.y += random.normal(0, PROCESS_NOISE_POSITION_M / TILE_SIZE_METERS)
  particle.heading += random.normal(0, PROCESS_NOISE_HEADING_DEG)
  ```
- Note: Convert velocity from m/s to tile units per second

### 8.4: update(measurements)
- measurements: list of dicts, each with:
  - 'position': (tile_x, tile_y)
  - 'heading': degrees
  - 'score': match quality score
- For each particle:
  ```
  # Find best measurement for this particle
  best_likelihood = 0
  for measurement in measurements:
      # Spatial likelihood
      dx_m = (particle.x - measurement.x) * TILE_SIZE_METERS
      dy_m = (particle.y - measurement.y) * TILE_SIZE_METERS
      dist = sqrt(dx_m² + dy_m²)
      spatial_likelihood = exp(-dist² / (2 * MEASUREMENT_NOISE_POSITION_M²))
      
      # Heading likelihood
      heading_diff = abs(angular_difference(particle.heading, measurement.heading))
      heading_likelihood = exp(-heading_diff² / (2 * MEASUREMENT_NOISE_HEADING_DEG²))
      
      # Combined likelihood
      likelihood = measurement.score * spatial_likelihood * heading_likelihood
      best_likelihood = max(best_likelihood, likelihood)
  
  # Update weight
  particle.weight *= best_likelihood
  
  # Normalize weights
  total_weight = sum(p.weight for p in particles)
  for p in particles:
      p.weight /= total_weight
  ```

### 8.5: resample()
- Check effective particle count: N_eff = 1 / Σ(weight²)
- If N_eff < RESAMPLE_THRESHOLD * NUM_PARTICLES:
  - Use systematic resampling (low-variance)
  - Generate NUM_PARTICLES new particles
  - Sample from current distribution (probability = weight)
  - Reset all weights to 1/NUM_PARTICLES
  - Add small jitter to prevent particle collapse:
    ```
    particle.x += random.normal(0, 10 / TILE_SIZE_METERS)  # 10m jitter
    particle.y += random.normal(0, 10 / TILE_SIZE_METERS)
    particle.heading += random.normal(0, 2)  # 2° jitter
    ```

### 8.6: get_estimate()
- Return weighted mean of particles
- Position: Weighted average of (x, y)
- Heading: Circular weighted mean
  ```
  # Convert headings to unit vectors
  mean_sin = sum(weight * sin(heading_rad) for ...)
  mean_cos = sum(weight * cos(heading_rad) for ...)
  mean_heading = atan2(mean_sin, mean_cos)
  ```
- Return: (x, y, heading)

### 8.7: get_uncertainty()
- Compute covariance of particle distribution
- Return: dict with:
  - 'position_std': Standard deviation in meters
  - 'heading_std': Standard deviation in degrees
  - '95_ellipse_axes': 2σ ellipse semi-major/minor axes

### 8.8: get_search_region()
- Compute 95% confidence ellipse from particles
- Return: dict with:
  - 'center': (tile_x, tile_y) mean position
  - 'radius_tiles': max(3σ, minimum_radius)
  - 'heading_mean': Mean heading
  - 'heading_range': ±(2σ or minimum_range)

### 8.9: check_divergence()
- Monitor particle spread
- If position_std > `DIVERGENCE_POSITION_THRESHOLD_M` (default 200m): return True (divergence detected)
- If max particle weight < `DIVERGENCE_WEIGHT_THRESHOLD` (default 0.01): return True
- Otherwise: return False

**Validation After Implementation**:
```python
from src.particle_filter import ParticleFilter, Particle
import numpy as np

# Initialize
pf = ParticleFilter(
    num_particles=100,
    initial_position=(55.7, 9.5),
    initial_heading=90,
    initial_spread={'position_meters': 50, 'heading_degrees': 10},
    zoom=16
)
print(f"✓ Initialized {len(pf.particles)} particles")

# Test predict
pf.predict(dt=0.2, velocity_mps=20.0, gyro_z_dps=5.0)
estimate = pf.get_estimate()
print(f"✓ After prediction: {estimate}")

# Test update
measurements = [{
    'position': estimate[:2],  # Near current estimate
    'heading': estimate[2],
    'score': 150.0
}]
pf.update(measurements)
print(f"✓ After update: weights sum = {sum(p.weight for p in pf.particles)}")

# Test resample
pf.resample()
print("✓ Resampling complete")

# Test search region
region = pf.get_search_region()
print(f"✓ Search region: center={region['center']}, radius={region['radius_tiles']}")

print("✓ All particle_filter validations passed")
```

---

## MODULE 9: src/temporal_searcher.py

**Purpose**: Temporal-aware best-first searcher using particle filter.

> **Integration note**: Steps 3–4 of `_process_frame_N` (formerly "Generate
> Focused Candidates" + "Focused Best-First Search") now delegate to
> `MetaTileBuilder.run()`. This replaces the direct priority-queue search
> with the two-pass 8-neighbour search + meta-tile construction pipeline.
> The particle filter update in Step 5 uses the top-3 tiles from
> `MetaTileBuilder` as its measurements.

**Implementation Requirements**:

### 9.1: Class TemporalSearcher

**Initialization**:
- `__init__(semantic_model, feature_matcher, tile_loader, config)`
- Store models and config
- Initialize `particle_filter = None`
- Initialize `frame_count = 0`
- Initialize `last_timestamp = None`
- Initialize `history = []`
- Instantiate `meta_tile_builder = MetaTileBuilder(feature_matcher, tile_loader, config)`
- Instantiate `semantic_confirmer = SemanticConfirmer(semantic_model, config)`

**Main Interface**:

### 9.2: process_frame(query_frame, imu_data, timestamp)
- imu_data: dict with:
  - 'lat', 'lon': IMU position estimate
  - 'heading': IMU heading estimate
  - 'pos_sigma': Position uncertainty (meters)
  - 'heading_sigma': Heading uncertainty (degrees)
  - 'velocity_mps': Velocity in m/s
  - 'gyro_z_dps': Yaw rate in deg/s

- If frame_count == 0:
  - Call `self._process_frame_0`
- Else:
  - Call `self._process_frame_N`

- Increment frame_count
- Update last_timestamp
- Append result to history
- Return: result dict

### 9.3: _process_frame_0(query_frame, imu_data, timestamp)
- **Frame 0 = Cold Start = Use Pipeline 1**
- Create BestFirstSearcher from Pipeline 1
- Run full search from IMU prior
- Get best match
- Initialize particle filter:
  - Center: Best match position
  - Spread based on match score:
    - High score (>150): 50m position, 10° heading
    - Medium score (100–150): 100m position, 20° heading
    - Low score (<100): 200m position, 30° heading
  - Create NUM_PARTICLES around this center
- Return: dict with:
  - 'position': (lat, lon)
  - 'heading': degrees
  - 'score': match score
  - 'tiles_tested': count
  - 'search_time': seconds
  - 'method': 'cold_start'

### 9.4: _process_frame_N(query_frame, imu_data, timestamp)
- **Subsequent Frames = Temporal Tracking + Two-Pass Meta-Tile Search**

**Step 1: Predict Particles**
- dt = timestamp - last_timestamp
- `particle_filter.predict(dt, imu_data['velocity_mps'], imu_data['gyro_z_dps'])`

**Step 2: Get Focused Search Region**
- `region = particle_filter.get_search_region()`
- Gives: center tile, search radius in tiles, heading range

**Step 3: Two-Pass Focused Search via MetaTileBuilder**
- Use particle region center as IMU prior for MetaTileBuilder:
  ```python
  center_lat, center_lon = tile_to_latlon(region['center'], zoom=16)
  meta_result = self.meta_tile_builder.run(
      query_frame=query_frame,
      imu_lat=center_lat,
      imu_lon=center_lon,
      search_radius_m=region['radius_tiles'] * TILE_SIZE_METERS,
      query_timestamp=timestamp
  )
  ```
- MetaTileBuilder performs:
  - First pass within particle region radius
  - Second pass: 8-neighbours of top-1 tile
  - Meta-tile construction from top-3
  - Meta-tile saved to `outputs/metatiles/metatile_{timestamp:.3f}.png`
  - Meta-tile verification (match count vs. `METATILE_MATCH_THRESHOLD`)

**Step 4: Extract Measurements for Particle Update**
- If `meta_result['verified']`:
  - Use `meta_result['top3_tiles']` as measurements (high confidence)
- If not verified:
  - Use first-pass top-1 tile only as single low-confidence measurement
  - Flag result confidence as LOW

**Step 5: Update Particle Filter**
- Convert top tiles to measurement format:
  ```python
  measurements = [
      {'position': (tx, ty), 'heading': imu_data['heading'], 'score': score}
      for tx, ty, score in meta_result['top3_tiles']
  ]
  ```
- `particle_filter.update(measurements)`
- `particle_filter.resample()`

**Step 6: Semantic Double-Confirmation**
- Run Branch A semantic segmentation on query frame (if not already done in parallel):
  ```python
  query_semantic_map = self.semantic_confirmer.segment(query_frame)
  ```
- Run centroid-based confirmation against meta-tile:
  ```python
  confirm_result = self.semantic_confirmer.confirm(
      query_semantic_map, meta_result['meta_tile']
  )
  ```
- Log confirmation confidence score

**Step 7: Get Final Estimate**
- `estimate = particle_filter.get_estimate()`
- Convert (tile_x, tile_y) to (lat, lon)
- Return: dict with:
  - 'position': (lat, lon)
  - 'heading': degrees
  - 'score': best match score from meta_result
  - 'tiles_tested': meta_result['first_pass_candidates']
  - 'search_time': seconds
  - 'method': 'temporal_tracking'
  - 'particle_spread': uncertainty in meters
  - 'n_eff': effective particle count
  - 'meta_tile_path': meta_result['meta_tile_path']
  - 'meta_tile_verified': meta_result['verified']
  - 'semantic_confidence': confirm_result['confidence']

**Step 8: Check for Divergence**
- If `particle_filter.check_divergence()`:
  - Log warning
  - Reinitialize with Frame 0 approach on next call
  - Reset `frame_count = 0`

### 9.5: get_trajectory()
- Return: list of all position estimates
- Format: [(lat, lon, heading, timestamp), ...]

### 9.6: save_trajectory(filepath)
- Save trajectory to CSV
- Columns: timestamp, lat, lon, heading, score, method, tiles_tested,
  meta_tile_verified, semantic_confidence

**Validation After Implementation**:
```python
from src.temporal_searcher import TemporalSearcher
from src import semantic_fingerprint, geometric_matcher
from src.meta_tile_builder import MetaTileBuilder
from config import config

model = semantic_fingerprint.load_semantic_model(config.SEMANTIC_MODEL_PATH, config.DEVICE)
matcher = geometric_matcher.initialize_matcher(config.DEVICE)
tile_loader = TileLoader(config.REFERENCE_TILES_DIR)

searcher = TemporalSearcher(model, matcher, tile_loader, config)
print("✓ TemporalSearcher initialized")

# Simulate Frame 0
query_frame_0 = ...  # Load actual frame
imu_data_0 = {
    'lat': 55.7, 'lon': 9.5, 'heading': 180,
    'pos_sigma': 100, 'heading_sigma': 5,
    'velocity_mps': 20, 'gyro_z_dps': 0
}
result_0 = searcher.process_frame(query_frame_0, imu_data_0, timestamp=0.523)
print(f"✓ Frame 0: {result_0['method']}, tiles={result_0['tiles_tested']}")

# Simulate Frame 1
query_frame_1 = ...  # Load next frame
imu_data_1 = {
    'lat': 55.7, 'lon': 9.5, 'heading': 180,
    'pos_sigma': 100, 'heading_sigma': 5,
    'velocity_mps': 20, 'gyro_z_dps': 0
}
result_1 = searcher.process_frame(query_frame_1, imu_data_1, timestamp=1.113)
print(f"✓ Frame 1: {result_1['method']}, tiles={result_1['tiles_tested']}")
print(f"✓ Meta-tile saved: {result_1['meta_tile_path']}")
print(f"✓ Semantic confidence: {result_1['semantic_confidence']:.3f}")

assert result_1['tiles_tested'] < result_0['tiles_tested']
print("✓ Temporal tracking is more efficient")

print("✓ All temporal_searcher validations passed")
```

---

## MODULE 10: src/meta_tile_builder.py

**Purpose**: Two-pass SuperPoint+LightGlue search, 8-neighbour expansion,
meta-tile construction, timestamped persistence, and meta-tile verification.

**Implementation Requirements**:

### 10.1: Class MetaTileBuilder

**Initialization**:
```python
def __init__(self, feature_matcher, tile_loader, config):
    self.matcher = feature_matcher
    self.tiles = tile_loader
    self.config = config
```

### 10.2: first_pass(query_frame, imu_lat, imu_lon, search_radius_m)
- Retrieve all tile candidates within `search_radius_m` of `(imu_lat, imu_lon)`
- Run `self.matcher.match(query_frame, tile_image)` for each candidate
- Return: ranked list of `(tile_x, tile_y, match_count)` descending

### 10.3: second_pass(query_frame, top_tile_x, top_tile_y)
- Compute 8 grid neighbours of `(top_tile_x, top_tile_y)` plus the pivot tile
  itself = 9 candidates total
- Run `self.matcher.match(query_frame, tile_image)` for each of the 9
- Rank by match count descending
- Return: top-3 as list of `(tile_x, tile_y, match_count)`

```python
NEIGHBOUR_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

def _get_neighbours(self, tx, ty):
    candidates = [(tx, ty)]  # Include pivot tile
    for dx, dy in NEIGHBOUR_OFFSETS:
        nx, ny = tx + dx, ty + dy
        if self.tiles.exists(nx, ny):
            candidates.append((nx, ny))
    return candidates
```

### 10.4: build_meta_tile(top3_tiles)
- Determine bounding box of the top-3 tile grid positions
- Create black canvas sized to fit all tiles at correct relative positions
- Paste each tile at its correct grid offset
- Return: meta_tile (numpy array, RGB)

```python
def build_meta_tile(self, top3_tiles):
    xs = [t[0] for t in top3_tiles]
    ys = [t[1] for t in top3_tiles]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    tile_px = 512
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    for tx, ty, _ in top3_tiles:
        img = self.tiles.load_aerial(tx, ty)
        col = tx - x_min
        row = ty - y_min
        canvas[row*tile_px:(row+1)*tile_px,
               col*tile_px:(col+1)*tile_px] = img
    return canvas
```

### 10.5: save_meta_tile(meta_tile, query_timestamp)
- Filename matches query frame naming convention exactly:
  `metatile_{query_timestamp:.3f}.png`
  (e.g. `frame_0.523.jpg` → `metatile_0.523.png`)
- Save to `config.METATILE_OUTPUT_DIR`

```python
def save_meta_tile(self, meta_tile, query_timestamp):
    import cv2
    fname = f"metatile_{query_timestamp:.3f}.png"
    out_path = self.config.METATILE_OUTPUT_DIR / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(meta_tile, cv2.COLOR_RGB2BGR))
    return out_path
```

### 10.6: verify_meta_tile(query_frame, meta_tile)
- Run `self.matcher.match(query_frame, meta_tile)`
- Return: `(match_count, match_result_dict)`
- Caller compares match_count against `config.METATILE_MATCH_THRESHOLD`

### 10.7: run(query_frame, imu_lat, imu_lon, query_timestamp,
           search_radius_m=None)
- Orchestrates full two-pass pipeline:

```python
def run(self, query_frame, imu_lat, imu_lon, query_timestamp, search_radius_m=None):
    if search_radius_m is None:
        search_radius_m = self.config.FIRST_PASS_SEARCH_RADIUS_M

    # Step 1: First pass
    first_pass_results = self.first_pass(query_frame, imu_lat, imu_lon, search_radius_m)
    if not first_pass_results:
        return None  # No tiles found in search region

    # Step 2: Second pass on 8-neighbours of top-1
    top1_tx, top1_ty, _ = first_pass_results[0]
    second_pass_results = self.second_pass(query_frame, top1_tx, top1_ty)
    top3 = second_pass_results[:self.config.METATILE_TOP_K]

    # Step 3: Build meta-tile
    meta_tile = self.build_meta_tile(top3)

    # Step 4: Save to disk (always, before verification)
    meta_tile_path = self.save_meta_tile(meta_tile, query_timestamp)

    # Step 5: Verify meta-tile against query frame
    match_count, match_result = self.verify_meta_tile(query_frame, meta_tile)
    verified = match_count >= self.config.METATILE_MATCH_THRESHOLD

    return {
        'meta_tile': meta_tile,
        'meta_tile_path': meta_tile_path,
        'top3_tiles': top3,
        'verification_matches': match_count,
        'verified': verified,
        'first_pass_candidates': len(first_pass_results),
    }
```

**Validation After Implementation**:
```python
from src.meta_tile_builder import MetaTileBuilder

builder = MetaTileBuilder(matcher, tile_loader, config)
result = builder.run(
    query_frame=query_frame,
    imu_lat=55.7,
    imu_lon=9.5,
    query_timestamp=0.523
)
assert result is not None
print(f"✓ Top-3 tiles: {result['top3_tiles']}")
print(f"✓ Meta-tile saved: {result['meta_tile_path']}")
print(f"✓ Verification matches: {result['verification_matches']}")
print(f"✓ Verified: {result['verified']}")
print(f"✓ First pass candidates: {result['first_pass_candidates']}")
print("✓ All meta_tile_builder validations passed")
```

---

## MODULE 11: src/semantic_confirmer.py

**Purpose**: Semantic double-confirmation via centroid-based feature matching
(MDPI paper method, doi:10.3390/rs17101671) between the query semantic map
(from Branch A) and the meta-tile semantic map (run after Stage 3 verification).

### 11.1: Class SemanticConfirmer

**Initialization**:
```python
def __init__(self, semantic_model, config):
    self.model = semantic_model
    self.config = config
```

### 11.2: segment(image)
- Apply 512×512 padding preprocessing (resize to 512×288, pad to 512×512)
- Run `self.model` inference
- Return: semantic mask (H×W, class indices 0–5)

### 11.3: extract_centroids(semantic_mask)
- For each class present in the mask:
  - Find connected components
  - Compute centroid (cx, cy) of each component
  - Record class label and component area
- Return: list of `{'class': int, 'cx': float, 'cy': float, 'area': int}`

### 11.4: match_centroids(query_centroids, reference_centroids)
- For each query centroid, find nearest reference centroid of same class
- A pair is matched if distance ≤ `config.CENTROID_MATCH_DISTANCE_THRESHOLD_PX`
- Score = number of matched pairs
- Return:
```python
{
    'matched_pairs': int,
    'total_query_centroids': int,
    'match_ratio': float,   # matched_pairs / total_query_centroids
    'confidence': float,    # match_ratio weighted by mean area of matched components
}
```
- Require at least `config.SEMANTIC_CONFIRM_MIN_PAIRS` matched pairs for
  non-zero confidence

### 11.5: confirm(query_semantic_map, meta_tile)
- **Do not re-run inference on query** — accept pre-computed `query_semantic_map`
  (already generated in Branch A; passing it here avoids duplicate inference)
- Segment `meta_tile` → meta-tile semantic map
- Extract centroids from both maps
- Run `match_centroids()`
- Return:
```python
{
    'matched_pairs': int,
    'match_ratio': float,
    'confidence': float,
    'query_centroids': int,
    'meta_tile_centroids': int,
}
```

**Validation After Implementation**:
```python
from src.semantic_confirmer import SemanticConfirmer

confirmer = SemanticConfirmer(semantic_model, config)

# Simulate Branch A: segment query frame once
query_semantic_map = confirmer.segment(query_frame)
print(f"✓ Query semantic map: {query_semantic_map.shape}, classes: {np.unique(query_semantic_map)}")

# Stage 4: confirm against meta-tile (no re-inference on query)
result = confirmer.confirm(query_semantic_map, meta_tile)
print(f"✓ Matched centroid pairs: {result['matched_pairs']}")
print(f"✓ Match ratio: {result['match_ratio']:.3f}")
print(f"✓ Confidence: {result['confidence']:.3f}")
print("✓ All semantic_confirmer validations passed")
```

---

## MODULE 12: src/trajectory_smoother.py

**Purpose**: Post-processing to smooth trajectory (optional enhancement).

> **Note**: This is the original Module 10 renumbered to Module 12 to
> accommodate the two new modules inserted above.

**Implementation Requirements**:

### 12.1: smooth_trajectory(positions, method='kalman')
- Input: list of (lat, lon, heading, timestamp)
- Methods:
  - 'kalman': Kalman smoother (forward-backward)
  - 'moving_average': Simple moving average
  - 'spline': Cubic spline interpolation
- Return: smoothed positions

### 12.2: detect_outliers(positions, threshold_meters)
- Detect position jumps > threshold
- Return: indices of outlier frames

### 12.3: fill_gaps(positions)
- If frames missing (gaps in timestamps)
- Interpolate positions

**Validation After Implementation**:
```python
from src.trajectory_smoother import smooth_trajectory, detect_outliers

# Test data: positions with noise
positions = [
    (55.7 + 0.0001*i + np.random.normal(0, 0.00005), 
     9.5 + 0.0001*i + np.random.normal(0, 0.00005), 
     180, i*0.2) 
    for i in range(20)
]

smoothed = smooth_trajectory(positions, method='kalman')
assert len(smoothed) == len(positions)
print("✓ Trajectory smoothing")

outliers = detect_outliers(positions, threshold_meters=100)
print(f"✓ Outlier detection: {len(outliers)} outliers")

print("✓ All trajectory_smoother validations passed")
```

---

## TESTING NOTEBOOK: notebooks/test_temporal_pipeline.ipynb

**Purpose**: End-to-end temporal pipeline testing.

### Cell 1: Setup
```python
from src.temporal_searcher import TemporalSearcher
from src import semantic_fingerprint, geometric_matcher
from src.meta_tile_builder import MetaTileBuilder
from src.semantic_confirmer import SemanticConfirmer
from config import config
import pandas as pd
import matplotlib.pyplot as plt
```

### Cell 2: Load Video Sequence
```python
# Load query frames (e.g., 50 frames)
query_frames = [load_frame(i) for i in range(50)]

# Load IMU data
imu_log = pd.read_csv("path/to/imu_gps_log_20260321_162024.csv")

# Load ground truth (for evaluation only)
ground_truth = imu_log[['latitude', 'longitude']].copy()
```

### Cell 3: Initialize Temporal Searcher
```python
model = semantic_fingerprint.load_semantic_model(config.SEMANTIC_MODEL_PATH, config.DEVICE)
matcher = geometric_matcher.initialize_matcher(config.DEVICE)
tile_loader = TileLoader(config.REFERENCE_TILES_DIR)
searcher = TemporalSearcher(model, matcher, tile_loader, config)
```

### Cell 4: Process All Frames
```python
results = []
for i, frame in enumerate(query_frames):
    imu_data = {
        'lat': imu_log.loc[i, 'latitude'],
        'lon': imu_log.loc[i, 'longitude'],
        'heading': imu_log.loc[i, 'heading'],
        'pos_sigma': 100,   # Use EKF covariance if extracted
        'heading_sigma': 15,
        'velocity_mps': np.sqrt(imu_log.loc[i, 'vel_n']**2 + imu_log.loc[i, 'vel_e']**2),
        'gyro_z_dps': imu_log.loc[i, 'gyro_z'] * (180 / np.pi)
    }
    
    timestamp = float(imu_log.loc[i, 'timestamp'])
    result = searcher.process_frame(frame, imu_data, timestamp=timestamp)
    results.append(result)
    
    print(f"Frame {i}: {result['method']}, {result['tiles_tested']} tiles, "
          f"{result['search_time']:.2f}s, verified={result.get('meta_tile_verified', 'N/A')}, "
          f"sem_conf={result.get('semantic_confidence', 0):.3f}")
```

### Cell 5: Analyze Performance
```python
frame_0_time = results[0]['search_time']
subsequent_times = [r['search_time'] for r in results[1:]]
mean_subsequent_time = np.mean(subsequent_times)

print(f"Frame 0 time: {frame_0_time:.2f}s")
print(f"Subsequent frames mean: {mean_subsequent_time:.2f}s")
print(f"Speedup: {frame_0_time / mean_subsequent_time:.1f}×")

frame_0_tiles = results[0]['tiles_tested']
subsequent_tiles = [r['tiles_tested'] for r in results[1:]]
mean_subsequent_tiles = np.mean(subsequent_tiles)
print(f"Frame 0 tiles: {frame_0_tiles}")
print(f"Subsequent frames mean: {mean_subsequent_tiles:.1f}")
print(f"Tile reduction: {frame_0_tiles / mean_subsequent_tiles:.1f}×")
```

### Cell 6: Compute Accuracy
```python
from src.tile_utils import haversine_distance

errors = []
for i, result in enumerate(results):
    est_lat, est_lon = result['position']
    gt_lat = ground_truth.loc[i, 'latitude']
    gt_lon = ground_truth.loc[i, 'longitude']
    error_m = haversine_distance(est_lat, est_lon, gt_lat, gt_lon) * 1000
    errors.append(error_m)

print(f"Mean error: {np.mean(errors):.1f}m")
print(f"Median error: {np.median(errors):.1f}m")
print(f"95th percentile: {np.percentile(errors, 95):.1f}m")
```

### Cell 7: Visualize Trajectory
```python
estimated_lats = [r['position'][0] for r in results]
estimated_lons = [r['position'][1] for r in results]
gt_lats = ground_truth['latitude'].values
gt_lons = ground_truth['longitude'].values

plt.figure(figsize=(12, 8))
plt.plot(gt_lons, gt_lats, 'g-', label='Ground Truth', linewidth=2)
plt.plot(estimated_lons, estimated_lats, 'r--', label='Estimated', linewidth=2)
plt.scatter(estimated_lons[0], estimated_lats[0], c='blue', s=100, marker='o', label='Start')
plt.scatter(estimated_lons[-1], estimated_lats[-1], c='orange', s=100, marker='s', label='End')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True)
plt.savefig('outputs/trajectories/trajectory_comparison.png', dpi=150)
plt.show()
```

### Cell 8: Error Timeline
```python
plt.figure(figsize=(14, 6))
plt.plot(errors, 'b-', linewidth=1.5)
plt.axhline(y=50, color='g', linestyle='--', label='50m threshold')
plt.axhline(y=100, color='orange', linestyle='--', label='100m threshold')
plt.xlabel('Frame Number')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.legend()
plt.grid(True)
plt.savefig('outputs/trajectories/error_timeline.png', dpi=150)
plt.show()
```

### Cell 9: Particle Spread Analysis
```python
spreads = [r['particle_spread'] for r in results[1:] if 'particle_spread' in r]

plt.figure(figsize=(14, 6))
plt.plot(spreads, 'r-', linewidth=1.5)
plt.axhline(y=200, color='red', linestyle='--', label='Divergence threshold')
plt.xlabel('Frame Number')
plt.ylabel('Particle Spread (m)')
plt.title('Particle Filter Uncertainty Over Time')
plt.legend()
plt.grid(True)
plt.savefig('outputs/trajectories/particle_spread.png', dpi=150)
plt.show()
```

### Cell 10: Performance Summary
```python
verified_count = sum(1 for r in results[1:] if r.get('meta_tile_verified', False))
sem_confidences = [r.get('semantic_confidence', 0) for r in results[1:]]

summary = {
    'Total frames': len(results),
    'Frame 0 time': f"{frame_0_time:.2f}s",
    'Mean subsequent time': f"{mean_subsequent_time:.2f}s",
    'Speedup': f"{frame_0_time / mean_subsequent_time:.1f}×",
    'Mean error': f"{np.mean(errors):.1f}m",
    'Median error': f"{np.median(errors):.1f}m",
    'Success rate (<50m)': f"{(np.array(errors) < 50).mean() * 100:.1f}%",
    'Success rate (<100m)': f"{(np.array(errors) < 100).mean() * 100:.1f}%",
    'Meta-tile verified rate': f"{verified_count / max(len(results)-1,1) * 100:.1f}%",
    'Mean semantic confidence': f"{np.mean(sem_confidences):.3f}",
}

for key, val in summary.items():
    print(f"{key:35s}: {val}")

import json
with open('outputs/trajectories/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

---

## UNIT TESTS

### test_particle_filter.py

```python
import pytest
from src.particle_filter import ParticleFilter, Particle
import numpy as np

def test_particle_initialization():
    pf = ParticleFilter(
        num_particles=100,
        initial_position=(55.7, 9.5),
        initial_heading=90,
        initial_spread={'position_meters': 50, 'heading_degrees': 10},
        zoom=16
    )
    assert len(pf.particles) == 100
    assert all(0 <= p.heading < 360 for p in pf.particles)

def test_predict():
    pf = ParticleFilter(100, (55.7, 9.5), 90, {'position_meters': 50, 'heading_degrees': 10}, 16)
    initial_x = pf.particles[0].x
    pf.predict(dt=1.0, velocity_mps=10.0, gyro_z_dps=5.0)
    assert pf.particles[0].x != initial_x

def test_update():
    pf = ParticleFilter(100, (55.7, 9.5), 90, {'position_meters': 50, 'heading_degrees': 10}, 16)
    estimate = pf.get_estimate()
    measurements = [{
        'position': estimate[:2],
        'heading': estimate[2],
        'score': 150.0
    }]
    pf.update(measurements)
    total_weight = sum(p.weight for p in pf.particles)
    assert abs(total_weight - 1.0) < 0.01

def test_resample():
    pf = ParticleFilter(100, (55.7, 9.5), 90, {'position_meters': 50, 'heading_degrees': 10}, 16)
    pf.particles[0].weight = 0.99
    for i in range(1, 100):
        pf.particles[i].weight = 0.01 / 99
    pf.resample()
    assert all(abs(p.weight - 0.01) < 0.001 for p in pf.particles)

def test_search_region():
    pf = ParticleFilter(100, (55.7, 9.5), 90, {'position_meters': 50, 'heading_degrees': 10}, 16)
    region = pf.get_search_region()
    assert 'center' in region
    assert 'radius_tiles' in region
    assert 'heading_mean' in region
    assert 'heading_range' in region
```

### test_temporal_searcher.py

```python
import pytest
from src.temporal_searcher import TemporalSearcher
from src import semantic_fingerprint, geometric_matcher
from config import config
import numpy as np

@pytest.fixture
def searcher():
    model = semantic_fingerprint.load_semantic_model(config.SEMANTIC_MODEL_PATH, config.DEVICE)
    matcher = geometric_matcher.initialize_matcher(config.DEVICE)
    tile_loader = TileLoader(config.REFERENCE_TILES_DIR)
    return TemporalSearcher(model, matcher, tile_loader, config)

def test_frame_0_processing(searcher):
    query_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    imu_data = {
        'lat': 55.7, 'lon': 9.5, 'heading': 180,
        'pos_sigma': 100, 'heading_sigma': 5,
        'velocity_mps': 20, 'gyro_z_dps': 0
    }
    result = searcher.process_frame(query_frame, imu_data, timestamp=0.523)
    assert result['method'] == 'cold_start'
    assert 'position' in result
    assert 'tiles_tested' in result

def test_temporal_tracking(searcher):
    query_frame_0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    imu_data_0 = {
        'lat': 55.7, 'lon': 9.5, 'heading': 180,
        'pos_sigma': 100, 'heading_sigma': 5,
        'velocity_mps': 20, 'gyro_z_dps': 0
    }
    result_0 = searcher.process_frame(query_frame_0, imu_data_0, timestamp=0.523)

    query_frame_1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    imu_data_1 = {
        'lat': 55.7, 'lon': 9.5, 'heading': 180,
        'pos_sigma': 100, 'heading_sigma': 5,
        'velocity_mps': 20, 'gyro_z_dps': 0
    }
    result_1 = searcher.process_frame(query_frame_1, imu_data_1, timestamp=1.113)

    assert result_1['method'] == 'temporal_tracking'
    assert 'meta_tile_path' in result_1
    assert 'semantic_confidence' in result_1
    assert result_1['tiles_tested'] <= result_0['tiles_tested']

def test_trajectory_saving(searcher, tmp_path):
    for i, ts in enumerate([0.523, 1.113, 1.642, 2.187, 2.741]):
        query_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        imu_data = {
            'lat': 55.7 + i*0.0001, 'lon': 9.5 + i*0.0001, 'heading': 180,
            'pos_sigma': 100, 'heading_sigma': 5,
            'velocity_mps': 20, 'gyro_z_dps': 0
        }
        searcher.process_frame(query_frame, imu_data, timestamp=ts)

    filepath = tmp_path / "trajectory.csv"
    searcher.save_trajectory(filepath)
    assert filepath.exists()
```

### test_meta_tile_builder.py

```python
import pytest
from src.meta_tile_builder import MetaTileBuilder
import numpy as np

@pytest.fixture
def builder(matcher, tile_loader, config):
    return MetaTileBuilder(matcher, tile_loader, config)

def test_first_pass_returns_sorted(builder):
    query_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    results = builder.first_pass(query_frame, 55.7, 9.5, search_radius_m=300)
    if len(results) > 1:
        assert results[0][2] >= results[1][2]  # Descending match count

def test_second_pass_9_candidates(builder):
    query_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    results = builder.second_pass(query_frame, 34500, 45030)
    assert len(results) <= 9  # At most 9 (fewer if near edge of tileset)

def test_build_meta_tile_shape(builder):
    top3 = [(34500, 45030, 80), (34501, 45030, 60), (34500, 45031, 50)]
    meta = builder.build_meta_tile(top3)
    assert meta.dtype == np.uint8
    assert meta.shape[2] == 3  # RGB

def test_save_meta_tile_filename(builder, tmp_path):
    builder.config.METATILE_OUTPUT_DIR = tmp_path
    meta = np.zeros((512, 512, 3), dtype=np.uint8)
    path = builder.save_meta_tile(meta, query_timestamp=0.523)
    assert path.name == "metatile_0.523.png"
    assert path.exists()

def test_run_returns_complete_dict(builder):
    query_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = builder.run(query_frame, 55.7, 9.5, query_timestamp=0.523)
    if result is not None:
        assert 'meta_tile' in result
        assert 'meta_tile_path' in result
        assert 'top3_tiles' in result
        assert 'verification_matches' in result
        assert 'verified' in result
        assert 'first_pass_candidates' in result
```

Run all tests:
```
pytest tests/test_particle_filter.py tests/test_temporal_searcher.py tests/test_meta_tile_builder.py -v
```

---

## CONFIGURATION ADDITIONS

Add to `config/config.py`:

```python
# ============================================================
# TEMPORAL PARTICLE FILTER CONFIGURATION
# ============================================================

NUM_PARTICLES = 100
PROCESS_NOISE_POSITION_M = 5.0
PROCESS_NOISE_HEADING_DEG = 2.0
MEASUREMENT_NOISE_POSITION_M = 50.0
MEASUREMENT_NOISE_HEADING_DEG = 10.0
RESAMPLE_THRESHOLD = 0.5

TEMPORAL_SEARCH_MAX_ITERATIONS = 50   # Smaller than Frame 0
TEMPORAL_MIN_SEARCH_RADIUS = 0.3      # At least 0.3 tiles (~100m)
TEMPORAL_MIN_ROTATION_RANGE = 10.0    # At least ±10°

DIVERGENCE_POSITION_THRESHOLD_M = 200.0
DIVERGENCE_WEIGHT_THRESHOLD = 0.01    # If max weight < this, reinitialize

PARTICLE_INIT_SPREAD_HIGH_CONF = {'position_meters': 50,  'heading_degrees': 10}
PARTICLE_INIT_SPREAD_MED_CONF  = {'position_meters': 100, 'heading_degrees': 20}
PARTICLE_INIT_SPREAD_LOW_CONF  = {'position_meters': 200, 'heading_degrees': 30}

# ============================================================
# TWO-PASS META-TILE CONFIGURATION
# ============================================================

FIRST_PASS_SEARCH_RADIUS_M = 300.0   # IMU uncertainty radius for first pass
SECOND_PASS_NEIGHBOURS = 8           # Always use all 8 neighbours of top-1
METATILE_TOP_K = 3                   # Top-K tiles to combine into meta-tile
METATILE_MATCH_THRESHOLD = 25        # Min inlier matches to accept meta-tile
                                     # (tune down if domain gap causes low counts)
METATILE_OUTPUT_DIR = Path("outputs/metatiles")
SEMANTIC_OUTPUT_DIR  = Path("outputs/semantic")
LOG_OUTPUT_DIR       = Path("outputs/logs")

# ============================================================
# SEMANTIC CONFIRMATION CONFIGURATION
# ============================================================

CENTROID_MATCH_DISTANCE_THRESHOLD_PX = 50  # Max pixel distance for centroid pair
SEMANTIC_CONFIRM_MIN_PAIRS = 3             # Min matched pairs for non-zero confidence
```

---

## ACCEPTANCE BENCHMARKS

Use these as engineering targets, not thesis claims:

- Subsequent frames should usually search fewer tiles than Frame 0
- Verification should fail gracefully rather than destabilize the trajectory
- Logged temporal results should beat or at least stabilize relative to IMU-only drift over a sequence
- Runtime should be reasonable for the present offline workflow first; embedded optimization comes later

A result is **not** good merely because it is fast. It is only good if the trajectory remains plausible and the failure cases are inspectable from logs.

---

## EXECUTION CHECKLIST FOR PIPELINE 3

1.  ✓ Pipeline 1 fully working and validated
2.  ✓ `particle_filter.py` implemented and tested (Module 8)
3.  ✓ `meta_tile_builder.py` implemented and tested (Module 10)
4.  ✓ `semantic_confirmer.py` implemented and tested (Module 11)
5.  ✓ `temporal_searcher.py` updated with MetaTileBuilder + SemanticConfirmer integration (Module 9)
6.  ✓ `trajectory_smoother.py` implemented and tested (Module 12)
7.  ✓ Config updated with all temporal + meta-tile + semantic parameters
8.  ✓ All output directories created: `outputs/metatiles/`, `outputs/semantic/`, `outputs/logs/`, `outputs/trajectories/`
9.  ✓ Unit tests passing (`test_particle_filter`, `test_temporal_searcher`, `test_meta_tile_builder`)
10. ✓ Frame 0 works identically to Pipeline 1
11. ✓ Frame 1+ faster than Frame 0 (verify timing)
12. ✓ Meta-tiles saved with correct `metatile_{ts:.3f}.png` naming
13. ✓ Meta-tile verification fires correctly against `METATILE_MATCH_THRESHOLD`
14. ✓ Semantic double-confirmation runs after verified meta-tile (no re-inference on query)
15. ✓ IMU fallback fires correctly when verification fails
16. ✓ Particle spread stable (<200m)
17. ✓ Trajectory smooth (no jumps)
18. ✓ Accuracy comparable to or better than Pipeline 1

---

## EXPECTED RESULTS

On a 50-frame video sequence:

```
Frame 0 (Cold Start):
  Method: cold_start
  Tiles tested: 23
  Search time: 4.2s
  Score: 234.5

Frame 1-49 (Temporal Tracking + Two-Pass):
  Method: temporal_tracking
  First pass candidates: 8-15
  Second pass tiles tested (per top-1): up to 9
  Meta-tile tiles combined: 3
  Search time: 0.4-0.9s (avg: 0.6s)
  Meta-tile verification matches: 25-80
  Semantic confirmation confidence: 0.3-0.8

Performance Summary:
  Speedup: 7.0× (4.2s → 0.6s)
  Tile reduction: 3.2× (23 → 7.3)
  Mean error: 32.4m
  Success rate (<50m): 78%
  Success rate (<100m): 94%
  Particle spread: 45-80m (stable)
  Meta-tile verified rate: ~70% (expect lower due to MSFS domain gap)
```

---

## DEBUGGING TIPS

**If particles diverge**:
- Check process noise (too high?)
- Check measurement noise (too low?)
- Verify IMU velocity and gyro_z are correct
- Check dt calculation (should be ~0.46s between frames)
- Visualize particle cloud over time

**If search still slow**:
- Verify particle-guided search uses region radius, not full 300m
- Check rotation range (should be ±10–20°, not ±30°)
- Profile code to find bottleneck
- Consider reducing NUM_PARTICLES to 50

**If meta-tile verification always fails**:
- Lower `METATILE_MATCH_THRESHOLD` (start at 15 for MSFS domain gap)
- Log raw match counts to calibrate threshold empirically
- MSFS → Danish government aerial is a hard domain gap; 25+ inliers
  is optimistic, 10–15 may be more realistic

**If semantic confirmation always low**:
- Log per-class centroid counts from both query and meta-tile masks
- Check that query frame preprocessing (512×512 pad) is consistent
- Seasonal mismatch (MSFS summer vs. early spring training data)
  will reduce class agreement — this is a known limitation

**If accuracy degrades**:
- Verify top-3 tiles used for particle update (not just top-1)
- Check particle weights computed correctly after update
- Check resampling not too aggressive
- Compare Frame 0 vs Frame 1+ accuracy separately

**If trajectory jumpy**:
- Enable trajectory smoother (Module 12)
- Check particle spread (if high, less trust in particles)
- Verify measurement likelihood calculation
- Consider Kalman smoothing post-processing

---

## PARAMETER TUNING GUIDE

**PROCESS_NOISE_POSITION_M**: Start 5.0m. Too confident → increase to 10m. Too uncertain → decrease to 3m.

**MEASUREMENT_NOISE_POSITION_M**: Start 50m. Particles ignore measurements → decrease to 30m. Particles jump with each measurement → increase to 80m.

**METATILE_MATCH_THRESHOLD**: Start 25. Tune down toward 10–15 if domain gap causes persistent verification failures. Tune up if false positives cause wrong pose estimates.

**CENTROID_MATCH_DISTANCE_THRESHOLD_PX**: Start 50px. Increase if semantic maps are misaligned due to scale difference between 512×288 query and larger meta-tile canvas.

**NUM_PARTICLES**: Start 100. Too slow → reduce to 50. Poor approximation → increase to 200.

**RESAMPLE_THRESHOLD**: Start 0.5. Too frequent resampling → decrease to 0.3. Particle degeneration → increase to 0.7.

---

## CRITICAL REMINDERS

1. **Build on Pipeline 1**: Don't rewrite, extend
2. **GPS is benchmarking only**: Never feed GPS back as a navigation input
3. **Seasonal model**: Semantic model trained on early spring / autumn /
   early winter Danish aerial imagery only — do not expect good predictions
   on MSFS summer scenes
4. **Domain gap is real**: SuperPoint+LightGlue match counts between MSFS
   query frames and Danish government reference tiles will be significantly
   lower than aerial-to-aerial — tune thresholds accordingly
5. **Branch A result reuse**: In Stage 4, pass the already-computed
   `query_semantic_map` from Branch A directly to `SemanticConfirmer.confirm()`
   — do not re-run inference on the query frame
6. **Meta-tile always saved**: Call `save_meta_tile()` before verification,
   not after — persist regardless of whether verification passes
7. **Test incrementally**: particle filter alone → meta_tile_builder alone →
   semantic_confirmer alone → temporal_searcher integration → full pipeline
8. **Validate Frame 0**: Must work identically to Pipeline 1
9. **Monitor particles**: Log spread, N_eff, weights every frame
10. **Save everything**: Meta-tiles, particle history, all estimates, timing
11. **Visualize**: Particle cloud, trajectory, error timeline, meta-tile composites
12. **Compare to Pipeline 1**: Accuracy should be similar or better

---

## FINAL HANDOFF RULES FOR THE AGENT

When implementing this document:

1. Start by reading the existing Pipeline 1 codebase and mapping reusable modules
2. Write the new modules with minimal, testable interfaces
3. Run module-level validation before moving to integration
4. Do not silently change interfaces across modules without updating all call sites
5. When uncertain, choose the simpler implementation that preserves observability
6. At the end, provide a short implementation report listing:
   - files added
   - files modified
   - tests passed
   - known limitations
   - next tuning priorities

---

END OF PIPELINE 3 INSTRUCTIONS
