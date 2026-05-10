# Phase A (remaining): PF Innovation Gate + Tile Plausibility + Result Writeback

## Context

GP8 is removed. Phase A items A1–A5 are **done** (gyro_y fix, r_used restructure, PF quality gate, EKF innovation gate, search radius cap). Phase B (homography root-cause fixes) is **deferred — do not start yet**.

Three Phase A items remain before the test run:
- PF innovation gate inside `temporal_searcher.py` (EKF gate in `run_pipeline.py` does not protect PF)
- Tile-center fallback plausibility filter (prevents meta_verified tile-centers from poisoning PF when they are 5 km away)
- Result dict writeback after `run_pipeline.py` innovation rejection (prevents trace.json vs results.csv disagreement)
- `max_visual_innovation_m` column in results.csv

After these four changes: run pytest, then stop. No Phase B yet.

GPS/sim lat/lon must never enter `ekf.update_position()` after initialization.

---

## Phase A (remaining) — exact changes

### Change 1 — PF innovation gate in `temporal_searcher.py`

**File**: `src/temporal_searcher.py` — Step 5 (lines ~431–466), inside `_process_frame_N()`

Before the `visual_gate_ok` block, add:
```python
homo_innovation_m = None
max_pf_innovation_m = max(
    150.0,
    3.0 * imu_data.get("pos_sigma", 0.0)
    + imu_data.get("velocity_mps", 0.0) * dt
    + 50.0
)
if homo_position is not None:
    homo_innovation_m = haversine_distance(
        homo_position[0], homo_position[1],
        imu_data["lat"], imu_data["lon"]
    )
visual_rejected_reason = ""
if (homo_position is not None
        and homo_innovation_m is not None
        and homo_innovation_m > max_pf_innovation_m):
    visual_rejected_reason = "pf_innovation_too_large"
```

Update `visual_gate_ok` to include innovation check:
```python
pf_innovation_ok = (
    homo_innovation_m is not None
    and homo_innovation_m <= max_pf_innovation_m
)
visual_gate_ok = (
    visual_quality.get("CShape", 0) > self.cfg.QUALITY_GATE_CSHAPE
    and visual_quality.get("inliers", 0) > self.cfg.QUALITY_GATE_INLIERS
    and homo_position is not None
    and homo_tile_pos is not None
    and pf_innovation_ok
)
```

Add `homo_innovation_m` and `max_pf_innovation_m` to the return dict at the bottom of `_process_frame_N()`.

### Change 2 — Tile-center plausibility filter in `temporal_searcher.py`

**File**: `src/temporal_searcher.py` — the `elif meta_result["verified"]:` and `else:` branches (~lines 446–466)

`tile_to_latlon` and `haversine_distance` are already imported. Use `max_pf_innovation_m` computed in Change 1.

Replace `elif meta_result["verified"]:` branch:
```python
elif meta_result["verified"]:
    plausible = []
    for tx, ty, score in meta_result["top3_tiles"]:
        tc_lat, tc_lon = tile_to_latlon(tx + 0.5, ty + 0.5, self.cfg.TMS_ZOOM_LEVEL)
        d = haversine_distance(tc_lat, tc_lon, imu_data["lat"], imu_data["lon"])
        if d <= max_pf_innovation_m:
            plausible.append((tx, ty, score))
    if plausible:
        pf_update_source = "tile_center"
        measurements = [
            {"position": (tx + 0.5, ty + 0.5),
             "heading": imu_data["heading"],
             "score": min(float(score), MAX_SCORE) / MAX_SCORE * 0.3}
            for tx, ty, score in plausible
        ]
    else:
        pf_update_source = "none"
        measurements = []
```

Replace `else:` (unverified top-1) branch:
```python
else:
    if meta_result["top3_tiles"]:
        tx, ty, score = meta_result["top3_tiles"][0]
        tc_lat, tc_lon = tile_to_latlon(tx + 0.5, ty + 0.5, self.cfg.TMS_ZOOM_LEVEL)
        d = haversine_distance(tc_lat, tc_lon, imu_data["lat"], imu_data["lon"])
        if d <= max_pf_innovation_m:
            pf_update_source = "tile_center"
            measurements = [
                {"position": (tx + 0.5, ty + 0.5),
                 "heading": imu_data["heading"],
                 "score": min(float(score), MAX_SCORE) / MAX_SCORE * 0.3}
            ]
        else:
            pf_update_source = "none"
            measurements = []
    else:
        pf_update_source = "none"
        measurements = []
```

Note: score * 0.3 applied to ALL tile-center fallbacks now (including the previously unscaled `verified` branch).

### Change 3 — Result dict writeback after innovation rejection in `run_pipeline.py`

**File**: `runtime/run_pipeline.py` — **both** `_process_one_frame()` (file mode, ~line 385) and `run_simconnect_mode()` (live mode, ~line 764)

Initialize `max_innovation_m = None` before the `if homo_pos is not None:` block in both paths.

After the innovation gate check (after the `if gate_pass and visual_innovation_m > max_innovation_m:` block), add writeback in both paths:
```python
result["gate_pass"] = gate_pass
result["visual_rejected_reason"] = visual_rejected_reason
result["visual_innovation_m"] = visual_innovation_m
result["max_visual_innovation_m"] = max_innovation_m
```
This keeps `trace.json` consistent with `results.csv` when the innovation gate fires.

### Change 4 — Add `max_visual_innovation_m` to results.csv

**File**: `runtime/run_pipeline.py`

In `RESULT_COLUMNS` (line ~74), add `"max_visual_innovation_m"` after `"visual_innovation_m"`:
```python
"visual_innovation_m", "max_visual_innovation_m", "visual_rejected_reason",
```

In both `result_row` dicts (file mode ~line 441, live mode ~line 821), add:
```python
"max_visual_innovation_m": round(max_innovation_m, 1) if max_innovation_m is not None else None,
```

### Change 5 — Fix NEXT_STEPS.md title

**File**: `Pipeline_3_Rev1/docs/NEXT_STEPS.md`

Change title from `# Next Steps: Post-Phase-A+B Roadmap` to `# Next Steps: Phase B and Beyond`.

Change preamble from:
> Describes work that follows the Phase A+B safety-gate and homography-quality fixes.

To:
> **Phase A is complete.** Phase B (homography quality root-cause fixes) is the immediate next step — see below. Phases C–H follow after Phase B produces stable live results.

Add a Phase B section at the top of the document (before Phase C) containing the 6 B-fixes from the plan file (B1–B6), each with file, description, and rationale. These are NOT yet implemented.

---

## Root cause analysis (Phase B — why homographies are wrong — FUTURE WORK)

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

## Phase A status (A1–A5 + A6 DONE; A7–A10 remaining — see above)

A1 (gyro_y), A2 (r_used restructure), A3 (PF quality gate), A4 (EKF innovation gate), A5 (search radius cap), A6 (diagnostic columns) — all implemented.

Remaining work is in "Phase A (remaining)" section above.

---

## Phase B: Root-cause homography quality fixes — FUTURE WORK

Do NOT implement until Phase A produces stable live results. These change algorithm behavior.

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

### pytest (run after all 5 changes)
```powershell
& .\.final_Pipeline_venv\Scripts\Activate.ps1
python -m pytest Pipeline_3_Rev1/tests -q
```
Expected: 26/37 pass (11 pre-existing failures in untouched code — `test_meta_tile_builder`, `test_semantic_confirmer`, `test_temporal_searcher`). No regressions vs current baseline.

### Static: RESULT_COLUMNS must include max_visual_innovation_m
```powershell
Select-String -Path Pipeline_3_Rev1/runtime/run_pipeline.py -Pattern 'max_visual_innovation_m'
```
Expect ≥ 4 hits (RESULT_COLUMNS + 2 result_row dicts + writeback).

### Stop here — user runs MSFS test next
After pytest passes, stop. No Phase B. No doc updates. User runs:
```powershell
python Pipeline_3_Rev1/runtime/run_pipeline.py --source simconnect --run-id live_022_a --max-frames 120
```

---

## What is NOT in scope here

- Phase B (homography fixes) — future work
- Folder cleanup / refactoring
- Thesis doc updates
- Semantic hard gate, meta_verified hard reject
- SAVE_PIPELINE_TRACE default=False
- File mode map/log mismatch (different dataset)
