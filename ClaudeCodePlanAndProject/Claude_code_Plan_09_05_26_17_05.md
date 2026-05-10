# Phase A+B: Post-GP8 Stability Fixes + Homography Quality Improvements

## Context

GP8 has been removed. Live mode coasts on IMU when `gate_pass=False`, confirmed by `ekf_before == ekf_after` on frame 101 of `live_021_Odense_f1`. The pipeline is now GPS-denied after init.

However, bad homography measurements and PF divergence are producing multi-hundred-metre or multi-kilometre visual deltas. This document covers both:
- **Phase A**: Safety gates (fast, low risk — no algorithm changes)
- **Phase B**: Root-cause fixes to the homography/measurement quality itself

GPS/sim lat/lon must never enter `ekf.update_position()` after initialization.

---

## Root cause analysis (Phase B — why homographies are wrong)

### Root cause 1: Degenerate homography from very few inliers
With 5 inliers and 8 DOF in a homography, the problem is underdetermined and the solution is numerically unstable. The projected center can end up anywhere. **Frame 101 had only 5 RANSAC inliers.** The quality gate (`QUALITY_GATE_INLIERS = 20`) should have caught this, but the PF step bypasses it (Problem 4 in Phase A).

### Root cause 2: Winner score ignores reprojection error
`_select_homography_winner()` in `visual_measurement.py:303–307`:
```python
score = n * CShape * convexity_bonus
```
`reproj_median` is computed but not used. A branch with 30 inliers and 15px median reproj error beats a branch with 20 inliers and 1px reproj. High reproj error means the matched points don't actually agree on the transformation — the H is fitting noise.

### Root cause 3: nadir_corrected uses wrong image dimensions
`visual_measurement.py:456`:
```python
f_px_approx = query_w / (2 * math.tan(math.radians(35)))
```
`query_w` here is the **resized rotated** query dimension (capped at 1280px, padded by rotation). The camera FOV should be computed from the **original** 1920px width. After heading rotation of e.g. 45°, the canvas expands to ~1750px; after resize to 1280, `query_w ≈ 1280` and `f_px_approx ≈ 914px` instead of the correct `≈1371px` from the original 1920px frame. This makes the nadir correction ~50% wrong in scale.

### Root cause 4: Centroid methods estimate feature cluster, not drone position
Methods B (`inlier_centroid`), C (`trimmed_centroid`), E (`weighted_centroid`) in `visual_measurement.py` compute the centroid of **matched reference keypoints**. This estimates the center of the matched region on the map, not the drone's camera ground point. If the drone is over a road and matches cluster along that road, the centroid is on the road, which may be far from the drone's nadir. These methods are systematically wrong for position estimation; they should be used only as last-resort fallbacks.

### Root cause 5: Cascade always starts with nadir_corrected
`temporal_searcher.py:550–552`:
```python
base = ["trimmed_centroid", "inlier_centroid", "weighted_centroid", "projected_center"]
return ["nadir_corrected"] + base
```
`nadir_corrected` is always tried first. Given root cause 3, this is propagating a wrong correction as the primary method. `projected_center` (Method A) — projecting the image center directly through H — is the cleanest estimator because it doesn't assume a camera model. The look-ahead correction in `run_pipeline.py` (110m backward along heading) already handles the systematic forward bias.

### Root cause 6: Meta-tile top-K produces sparse, irregular layout
`meta_tile_builder.py:320` takes `top_k = second_pass_results[:METATILE_TOP_K]`. The top-3 tiles by match count may be geometrically scattered (e.g. top-left + top-right + bottom-right corner of a 3×3 grid). The meta-tile canvas then has black gaps. `check_in_black()` in `visual_measurement.py:380–388` correctly catches these, but the cascade falls through to centroid methods (root cause 4) which are also unreliable.

### Root cause 7: No minimum inlier ratio check
If LightGlue produces 100 raw matches but RANSAC keeps only 8 (8% ratio), the few correct inliers are buried in many incorrect ones. The H is not trustworthy. There is currently no ratio check — only a count check.

### Root cause 8: PF search radius unbounded after divergence
`particle_filter.py:251`: `radius_m = max(3 * unc["position_std_m"], min_radius_m)`. Frame 101: spread=1624m → radius=4845m. A 4845m radius at zoom 16 covers ~200 tiles. The first pass tests all of them; the semantic filter keeps top-10; but these 10 tiles span a huge area and are unlikely to contain the correct tile. The H from such a match is garbage.

---

## Step 0 — Create NEXT_STEPS.md (FIRST, before any code changes)

**File to create**: `Pipeline_3_Rev1/docs/NEXT_STEPS.md`

This document captures the full roadmap for what comes after Phase A+B is executed. Content: Phase C (measurement A/B diagnostics), Phase D (meta-tile and ranking improvements), Phase E (EKF/PF integration tightening), Phase F (robustness), Phase G (production), Phase H (thesis). Include for each phase: motivation, specific files to change, and acceptance criteria. This file is created before any code edits so the user has the forward roadmap before approving any implementation.

---

## Critical files

| File | Role |
|---|---|
| `runtime/run_pipeline.py:308–430` | `_process_one_frame()` file mode |
| `runtime/run_pipeline.py:660–800` | `run_simconnect_mode()` live mode |
| `src/temporal_searcher.py:278–552` | `_process_frame_N()`, cascade, PF update |
| `src/visual_measurement.py:286–315` | Winner selection |
| `src/visual_measurement.py:322–495` | Measurement extraction (all methods) |
| `src/meta_tile_builder.py:278–351` | Two-pass search + meta-tile assembly |
| `src/particle_filter.py:244–260` | `get_search_region()` — no upper bound |
| `config/config.py` | All thresholds |

---

## Phase A: Safety gates (minimal, low risk)

Apply in order. No algorithm changes — only gating and logging.

### A1 — Fix gyro axis: PF yaw-rate is wrong
**Files**: `run_pipeline.py:335` (file mode) and `run_pipeline.py:702` (live mode)

`ekf_ins.py:649–652` establishes: `gyro_y` = MSFS Y (up) = standard Z = **yaw**. Currently both paths pass `gyro_z`.

```python
# Before (both lines):
"gyro_z_dps": row.get("gyro_z", 0.0) * (180.0 / math.pi)

# After (both lines):
"gyro_z_dps": row.get("gyro_y", 0.0) * (180.0 / math.pi)
```
Key name stays `"gyro_z_dps"` for now; rename is a separate cleanup.

### A2 — Fix r_used_sqrt misleading on gate-fail in live mode
**File**: `run_pipeline.py:720–750` (live mode only)

Live mode computes `r_used` inside `if homo_pos is not None:` regardless of `gate_pass`, then gates only the EKF call. File mode (lines 376–386) correctly has:
```python
r_used = None
if gate_pass and homo_pos is not None:
    r_used = ...
    ekf.update_position(...)
```
Restructure live mode to match this shape exactly.

### A3 — Gate PF from bad homography
**File**: `temporal_searcher.py:415–444`

```python
# Before: unconditional homo_tile_pos check
if homo_tile_pos is not None:
    measurements = [...]

# After: gate on quality
visual_gate_ok = (
    visual_quality.get("CShape", 0) > self.cfg.QUALITY_GATE_CSHAPE
    and visual_quality.get("inliers", 0) > self.cfg.QUALITY_GATE_INLIERS
    and homo_position is not None
    and homo_tile_pos is not None
)
if visual_gate_ok:
    pf_update_source = "homography"
    measurements = [{"position": homo_tile_pos, ...}]
elif meta_result["verified"]:
    pf_update_source = "tile_center"
    measurements = [...]
else:
    pf_update_source = "none"
    measurements = []
```

### A4 — EKF innovation gate
**Files**: Both `_process_one_frame()` and `run_simconnect_mode()`, after lookahead correction, before `ekf.update_position()`.

Import needed: `from src.tile_utils import haversine_distance` (add to `run_pipeline.py` imports; already in `temporal_searcher.py`).

```python
visual_innovation_m = haversine_distance(
    homo_pos[0], homo_pos[1], ekf_lat, ekf_lon
)
pos_sigma_now = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))
max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * 1.0 + 50.0)

visual_rejected_reason = ""
if visual_innovation_m > max_innovation_m:
    gate_pass = False
    visual_rejected_reason = "innovation_too_large"
```
Log both values. The `vel * 1.0` term uses 1.0s as a conservative dt in live mode where exact dt isn't wired in yet.

### A5 — PF search radius cap
**File**: `config/config.py` — add:
```python
MAX_TEMPORAL_SEARCH_RADIUS_M = 1500.0
```
**File**: `temporal_searcher.py:297–300` — after computing `search_radius_m`:
```python
search_radius_capped = False
if search_radius_m > self.cfg.MAX_TEMPORAL_SEARCH_RADIUS_M:
    search_radius_m = self.cfg.MAX_TEMPORAL_SEARCH_RADIUS_M
    search_radius_capped = True
    logger.warning("PF search radius capped at %.0f m (was %.0f m)",
                   self.cfg.MAX_TEMPORAL_SEARCH_RADIUS_M, uncapped_radius)
```

### A6 — Add diagnostic columns to results.csv
Propagate from `_process_frame_N()` result dict, through `run_pipeline.py` result_row:
- `visual_innovation_m` — distance between corrected homo_pos and ekf_lat/lon
- `visual_rejected_reason` — "" / "innovation_too_large" / "quality_gate" / "none"
- `pf_update_source` — "homography" / "tile_center" / "none"
- `search_radius_m` — actual radius used (after cap)
- `search_radius_capped` — 1/0

---

## Phase B: Root-cause homography quality fixes

Apply after Phase A produces stable runs. These change algorithm behavior.

### B1 — Fix nadir_corrected FOV to use original camera dimensions
**File**: `visual_measurement.py:322–358` (function signature and line 456)

Add parameter `original_query_w: int` (default 1920) to `extract_visual_measurements()`. Use it for the focal length approximation:
```python
# Before:
f_px_approx = query_w / (2 * math.tan(math.radians(35)))

# After:
f_px_approx = original_query_w / (2 * math.tan(math.radians(35)))
```
Pass `original_query_w = query_frame.shape[1]` (the un-rotated, un-resized query width) from `temporal_searcher._process_frame_N()` when calling `extract_visual_measurements`.

Also update the call sites in `_process_frame_0` (cold start).

### B2 — Add reprojection error penalty to winner score
**File**: `visual_measurement.py:303–307` (`_select_homography_winner`)

```python
# Before:
def score(branch):
    return branch["inliers"] * branch["CShape"] * (1.5 if branch["convex"] else 1.0)

# After:
def score(branch):
    reproj_penalty = 1.0 + branch["reproj_median"]
    return branch["inliers"] * branch["CShape"] * (1.5 if branch["convex"] else 1.0) / reproj_penalty
```

Also add hard rejection: if `CShape < QUALITY_GATE_CSHAPE` or `reproj_median > MAX_REPROJ_PX`, return `(None, None, None)` for that branch regardless of score. Add `MAX_REPROJ_PX = 8.0` to config.

### B3 — Add minimum inlier ratio check to winner selection
**File**: `visual_measurement.py:286–315`

After RANSAC, compute `inlier_ratio = inliers / n_correspondences`. If `inlier_ratio < MIN_INLIER_RATIO`, mark branch as invalid. Add `MIN_INLIER_RATIO = 0.10` to config.

Rationale: 100 raw matches → 5 inliers (5%) means the matcher found mostly noise. A 10% ratio floor is conservative.

### B4 — Change cascade order: projected_center first
**File**: `temporal_searcher.py:540–552` (`_build_cascade`)

```python
# Before:
return ["nadir_corrected"] + ["trimmed_centroid", "inlier_centroid", "weighted_centroid", "projected_center"]

# After:
base = ["projected_center", "trimmed_centroid", "inlier_centroid", "weighted_centroid"]
# nadir_corrected only when pitch or roll is substantial and camera model is validated
pitch_large = abs(pitch_rad) > 0.10   # ~6 degrees
roll_large  = abs(roll_rad) > 0.10
if pitch_large or roll_large:
    return ["nadir_corrected"] + base
return base
```

The look-ahead correction in `run_pipeline.py` already removes the systematic forward bias. `nadir_corrected` adds a small secondary correction based on an unvalidated FOV model — it should only be attempted when attitude angles are large enough to justify it, and after B1 fixes the scale.

### B5 — Use consistent 3×3 meta-tile patch instead of arbitrary top-K
**File**: `meta_tile_builder.py:277–351` (the `run()` orchestration)

After second pass identifies the best tile (`top1_from_second_pass`), instead of `second_pass_results[:3]`, build a 3×3 grid patch:
```python
best_tx, best_ty = top1_tx, top1_ty  # from second pass winner
grid_tiles = []
for dx in range(-1, 2):
    for dy in range(-1, 2):
        tx, ty = best_tx + dx, best_ty + dy
        if self.tiles.exists(tx, ty):
            score = match_scores.get((tx, ty), 0)  # reuse second-pass scores
            grid_tiles.append((tx, ty, score))
```
This gives a rectangular canvas (max 3×3=9 tiles = 1536×1536px) with no arbitrary gaps. The `px_to_latlon` function already handles rectangular tile layouts. Pass `grid_tiles` as `top3_tiles` to `build_meta_tile` and `extract_visual_measurements`.

**Config**: Rename `METATILE_TOP_K` to `METATILE_PATCH_SIZE` = 3 (meaning 3×3 grid). The constant is no longer "top K by score" but "±1 neighbourhood radius".

### B6 — Add scale consistency check on homography projected quad
**File**: `compute_shape_confidence()` in `visual_measurement.py:73–159`

After computing `proj_area`, check scale reasonableness. The projected quad should cover a fraction of the meta-tile that makes geometric sense given the drone altitude. At zoom 16, GSD ≈ 0.65m/px. At 300m AGL with ~70° FOV, the footprint is ≈400m×250m ≈ 615×385px on the reference. Query image is ~1280px wide after resize. So projected quad width should roughly be 300–800px on the meta-tile (512–1536px wide).

Add to `CShape` dict:
```python
"scale_ok": 0.01 * expected_area < proj_area < 10 * expected_area
```
If `scale_ok == False`, treat the branch as degenerate (set CShape to 0).

---

## Verification

### Phase A static checks
```powershell
# No GPS in EKF update path
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'R_pos_m2=200'
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'sim_lat|sim_lon'

# r_used must be inside gate_pass block in both paths
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'r_used' -Context 0,2

# gyro_y in both imu_data dicts
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'gyro_z_dps.*gyro_y'
```

### Phase A pytest
```powershell
& .\.final_Pipeline_venv\Scripts\Activate.ps1
python -m pytest Pipeline_3_Rev1/tests -q
```
All 6 tests must pass.

### Phase A behavioural (live, 120 frames)
```powershell
python Pipeline_3_Rev1/runtime/run_pipeline.py --source simconnect --run-id live_022_a --max-frames 120
```
Check in results.csv:
- Gate-fail frames: `r_used_sqrt = None`
- `visual_innovation_m` present, < 200m for gate-pass frames
- `pf_update_source` never "homography" when `gate_pass = 0`
- `search_radius_capped = 1` with warning logged when PF spreads
- No `final_lat`/`final_lon` jump > 500m in one frame

### Phase B behavioural (after each B fix, re-run live 120 frames)
For each B fix, compare:
- Gate-pass rate (want ≥ baseline 96%)
- Mean/median `visual_innovation_m` on accepted frames (want < 100m)
- `selected_measurement_method` distribution (want "projected_center" dominant)
- `pos_sigma` over time (want bounded, not monotonically growing)

---

## What is NOT in scope here

- Folder cleanup / refactoring
- Thesis doc updates
- Semantic hard gate (Phase C — after diagnostics confirm B is stable)
- meta_verified hard reject (Phase C)
- SAVE_PIPELINE_TRACE default=False (separate config change)
- File mode map/log mismatch (separate dataset issue)
- Heading initialization magnetic/true consistency (Phase D — logging shows it's probably fine)
