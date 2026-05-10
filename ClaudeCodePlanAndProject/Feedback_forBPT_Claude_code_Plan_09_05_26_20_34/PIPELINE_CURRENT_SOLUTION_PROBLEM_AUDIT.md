# Pipeline 3 Current Solution — Problem Audit README

**Scope:** GPS-denied UAV localization pipeline using SimConnect/MSFS input, EKF/INS propagation, semantic segmentation, SuperPoint+LightGlue feature matching, meta-tile search, homography-based visual localization, particle filtering, and EKF visual-position fusion.

**Source set reviewed:**

- `runtime/run_pipeline.py`
- `runtime/simconnect_adapter.py`
- `config/config.py`
- `src/best_first_search.py`
- `src/ekf_ins.py`
- `src/geometric_matcher.py`
- `src/image_utils.py`
- `src/meta_tile_builder.py`
- `src/particle_filter.py`
- `src/position_estimator.py`
- `src/semantic_confirmer.py`
- `src/semantic_model.py`
- `src/semantic_tile_scorer.py`
- `src/temporal_searcher.py`
- `src/tile_utils.py`
- `src/visual_measurement.py`
- `src/wmm_declination.py`
- uploaded run evidence: `results.csv`, trace JSONs, IMU JSONs, `terminal_output.txt`, `live_analysis.pdf`, `pipeline_trace.pdf`

---

## 0. Executive verdict

The pipeline is **not broken**, but it is **not clean enough to be called robust yet**.

It produces plausible localization output, but the current implementation has three major weaknesses:

1. **The live runtime path and file replay path duplicate the same fusion logic.** This is dangerous because future fixes can easily be applied to one path and missed in the other.
2. **Several geometry conversions and visual-measurement assumptions are inconsistent.** The worst issue is the mismatch between old `position_estimator.py` pixel-to-TMS conversion and the newer `visual_measurement.py` conversion.
3. **The visual quality gate is too weak.** The uploaded run proves that frames with high inlier counts and acceptable CShape can still be wrong by more than 150 m.

The system is thesis-usable as a research prototype, but it should be described honestly as an **experimental GPS-denied localization prototype**, not as a flight-ready navigation system.

> **[Claude Code review]:** The executive verdict is accurate. One update since this review was written: Phase R (relocalization recovery after innovation rejection streak) has been implemented. It adds `visual_quality_pass`, `ekf_update_applied`, `relocalization_candidate`, `relocalization_applied` columns, the `vel * dt_gate` fix (clamped [0.5, 4.0]s), covariance inflation before recovery update, and PF cold-start reset on recovery. The `ver_matches` NameError crash mentioned in CP-01 is also fixed via `_extract_meta_quality()`. The three weaknesses listed remain valid for everything outside Phase R's scope.

---

## 1. Uploaded run evidence snapshot

The uploaded `results.csv` contains **131 frames**.

| Metric | Value |
|---|---:|
| Frames | 131 |
| Gate pass frames | 120 / 131 |
| Gate pass rate | 91.6% |
| Mean online EKF error | 72.4 m |
| Median online EKF error | 59.1 m |
| Maximum online EKF error | 194.8 m |
| Mean inference latency | 2416 ms |
| Median inference latency | 2434 ms |
| P90 inference latency | 2712 ms |
| Maximum inference latency | 3305 ms |

Worst high-error accepted frames from the uploaded `results.csv`:

| Frame | Error [m] | CShape | Inliers | Gate | EKF update | Visual innovation [m] | Max innovation [m] | R std [m] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 194.8 | 0.591 | 618 | PASS | yes | 181.1 | 238.9 | 27.6 |
| 114 | 189.6 | 0.467 | 567 | PASS | yes | 77.0 | 262.1 | 79.4 |
| 115 | 179.3 | 0.412 | 406 | PASS | yes | 136.4 | 228.4 | 76.5 |
| 116 | 175.0 | 0.464 | 498 | PASS | yes | 95.0 | 252.1 | 77.9 |
| 103 | 167.1 | 0.690 | 396 | PASS | yes | 131.4 | 256.4 | 27.0 |

**Interpretation:** the current gate is good at rejecting obviously bad homographies, but it is not strong enough to reject plausible-looking but spatially wrong homographies. That is the central technical weakness.

---

## 2. Severity scale

| Level | Meaning |
|---|---|
| Critical | Can corrupt localization, hide a wrong estimate, or break future maintenance badly. |
| High | Can produce wrong behavior under realistic conditions or make debugging unreliable. |
| Medium | Does not necessarily break the run, but creates fragility or misleading outputs. |
| Low | Cleanup/readability/consistency issue. |

---

# 3. Confirmed problems

These are directly visible from the uploaded source files or from the uploaded run outputs.

---

## CP-01 — Live mode and file mode duplicate the same visual-fusion logic

**Severity:** Critical  
**Files:** `run_pipeline.py`

### Evidence

- `_process_one_frame()` implements file-mode per-frame logic in `run_pipeline.py:364-591`.
- `run_simconnect_mode()` re-implements the same logic inline in `run_pipeline.py:774-1181`.
- The duplicated sections include:
  - IMU dict construction
  - `TemporalSearcher.process_frame()` call
  - look-ahead correction
  - innovation gate
  - relocalization logic
  - adaptive visual-measurement covariance `R`
  - EKF `update_position()`
  - result-row construction
  - optional output writing

### Why this matters

This is how bugs survive. A fix can be applied to file mode but missed in live mode, or the opposite. You already saw this pattern with the `ver_matches` issue.

### Suggested fix

Extract a shared function, for example:

```python
def process_runtime_visual_update(
    *,
    frame_idx: int,
    timestamp: float,
    query_frame: np.ndarray,
    row_dict: dict,
    ekf: ErrorStateEKF,
    searcher: TemporalSearcher,
    recovery_state: dict,
    prev_frame_ts: float | None,
    image_name: str,
    inference_ms: float | None,
) -> RuntimeFrameResult:
    ...
```

Then both file mode and SimConnect mode call the same function.

### Validation test

Run these two paths over the same saved frame/IMU sequence:

```bash
python runtime/run_pipeline.py --source file --run-id test_file --max-frames 30
python runtime/run_pipeline.py --source simconnect-replay --run-id test_live_replay --max-frames 30
```

The two `results.csv` files should match within numerical tolerance.

> **[Claude Code review]:** Confirmed. The `ver_matches` NameError bug proved this exactly — the fix had to be applied to both modes separately via `_extract_meta_quality()`. The current file mode uses `_process_one_frame()` (a helper) while live mode is fully inline, so even a partial refactor where `_process_one_frame` absorbs the live path would eliminate the duplication. The Phase R recovery logic already had to be written twice (once in `_process_one_frame` and once inline in `run_simconnect_mode`) — another direct demonstration of the cost.

---

## CP-02 — `run_simconnect_mode()` is too large to be safe

**Severity:** High  
**File:** `run_pipeline.py`

### Evidence

- `run_simconnect_mode()` spans `run_pipeline.py:774-1181`, roughly 408 lines.
- It handles SimConnect connection, EKF bootstrap, row polling, repeated EKF prediction, frame acquisition, visual processing, innovation gating, relocalization, EKF update, CSV writing, PX4 output writing, trace writing, timing writing, and shutdown.

### Why this matters

This function is the live flight path. A live flight path should be boring, short, and easy to audit. This one is not. It is too easy to introduce a hidden bug while changing unrelated logging or optional output code.

### Suggested fix

Split into helpers:

| Helper | Responsibility |
|---|---|
| `_bootstrap_live_ekf(source)` | Wait for valid sample and initialize EKF. |
| `_step_ekf_on_new_row(source, ekf, prev_ts, last_imu_ts)` | EKF predict/update only when row timestamp changes. |
| `_get_new_frame(source, last_frame_id)` | Return only new frames. |
| `_process_runtime_frame(...)` | Shared visual + EKF fusion logic. |
| `_write_outputs(...)` | CSV, PX4, trace, timing outputs. |
| `_close_optional_outputs(...)` | Safe cleanup. |

### Validation test

Before refactor, save a 20-frame `results.csv`. After refactor, run the same input and compare:

```python
assert max(abs(old.final_lat - new.final_lat)) < 1e-9
assert max(abs(old.final_lon - new.final_lon)) < 1e-9
```

> **[Claude Code review]:** Confirmed. The 408-line function mixes three concerns: SimConnect/EKF bootstrapping, the per-frame algorithm, and optional output writing. These are independently testable and independently changeable — they should be separated. Solving CP-01 (shared fusion core) naturally splits out the algorithm piece, which leaves the live-mode function as bootstrap + loop + output, still ~200 lines but much safer.

---

## CP-03 — File mode double-processes the first EKF row

**Severity:** High  
**Files:** `run_pipeline.py`, `simconnect_adapter.py`

### Evidence

- `_init_ekf(raw_df, start_row)` loops through `range(start_row + 1)` and calls `step_ekf()` on every row up to and including `start_row`.
- `FileSource.iter_aligned(start_row, ...)` then starts yielding from the same `start_row`.
- `_process_one_frame()` calls `step_ekf()` again on that same CSV row.

### Why this matters

At `start_row=0`, the first row is processed once during EKF warm-up and again during the first frame. Even if `dt=0`, the measurement updates still run. That can artificially reduce covariance and make the filter look more confident than it should be.

### Suggested fix

Change `_init_ekf()` so it warms only through `start_row - 1`:

```python
for i in range(start_row):
    row_dict = raw_df.iloc[i].to_dict()
    step_ekf(ekf, row_dict, prev_ts)
    prev_ts = row_dict["timestamp"]
```

Or keep `_init_ekf()` unchanged and start `iter_aligned()` from `start_row + 1`. The first option is cleaner.

### Validation test

Add a unit test with a mock EKF that counts calls to `step_ekf()`. For `start_row=0`, the first frame should result in exactly one call, not two.

> **[Claude Code review]:** Confirmed. Traced through the code: `_init_ekf(raw_df, 0)` loops `range(1)` = [0], processing row 0 and returning `prev_ts = row_0_timestamp`. Then `iter_aligned(0, ...)` starts at index 0, yielding row 0 first. `_process_one_frame` calls `step_ekf(ekf, row_0, prev_ts=row_0_timestamp)` — so `dt = row_0_ts - row_0_ts = 0`. At dt=0 the IMU propagation does nothing (no position/velocity change), but the barometric, magnetometer, and airspeed measurement updates still apply their Kalman corrections, shrinking P twice from the same data. The effect is P underestimation by the first frame. The suggested fix (change `range(start_row + 1)` to `range(start_row)`) is correct and safe.

---

## CP-04 — SimConnect frame and row are not returned atomically

**Severity:** High  
**Files:** `simconnect_adapter.py`, `run_pipeline.py`

### Evidence

- `SimConnectLiveSource.get_latest_row()` returns the latest row.
- `SimConnectLiveSource.get_latest_frame()` separately returns the latest frame, frame ID, and capture timestamp.
- `run_simconnect_mode()` calls these separately.
- The background thread updates rows more often than frames.

### Why this matters

The visual frame and the IMU/GPS row can be slightly time-misaligned. At aircraft speeds around 60-70 m/s, even a few hundred milliseconds matters. At 0.2 s mismatch, the aircraft has already moved roughly 12-14 m.

### Suggested fix

Store the row together with the frame at capture time:

```python
self._latest_frame_packet = {
    "img": img_rgb,
    "frame_id": self._frame_id,
    "capture_ts_perf": t_start,
    "row_at_capture": copy.copy(row),
}
```

Then return this packet from a single method:

```python
def get_latest_frame_packet(self):
    return img, frame_id, capture_ts, row_at_capture
```

### Validation test

Log `row_at_capture["timestamp"]` and the row used for EKF visual update. The difference should be near zero for the visual frame.

> **[Claude Code review]:** Partially confirmed, with a correction to the description. Looking at the actual background thread code: when a frame IS captured, `_latest_row`, `_latest_img`, `_frame_id`, and `_latest_frame_capture_ts` are all updated atomically inside the same `with self._lock` block. So the row stored AT frame capture time is the simultaneously-captured row. The problem is different from what the reviewer describes: the main thread calls `get_latest_row()` (one lock acquisition) and then `get_latest_frame()` (a second, separate lock acquisition). Between these two calls, the background thread can have pushed a newer IMU row without capturing a new frame. So the main thread ends up using a row that is 1–4 IMU intervals newer than the row that was current at frame capture. At 50Hz IMU target (actual ~7Hz), this can be up to ~150ms and ~10m at cruise speed. The suggested fix of bundling `row_at_capture` with the frame is the right solution and avoids this race.

---

## CP-05 — The particle filter search region is computed but its center is ignored

**Severity:** Critical  
**File:** `temporal_searcher.py`

### Evidence

In `_process_frame_N()`:

```python
region = self.particle_filter.get_search_region()
center_lat, center_lon = imu_data["lat"], imu_data["lon"]
search_radius_m = max(region["radius_tiles"] * self.cfg.TILE_SIZE_METERS, ...)
```

The code uses the PF region radius, but centers the meta-tile search on the EKF/IMU position, not on the PF estimate.

### Why this matters

The file says the temporal search is particle-guided, but the search center is not particle-guided. If the EKF drifts but the PF estimate is better, the search still follows the EKF. This weakens the entire purpose of the particle filter.

### Suggested fix

Use `region["center"]` from `ParticleFilter.get_search_region()`:

```python
cx_tile, cy_tile = region["center"]
center_lat, center_lon = tile_to_latlon(cx_tile, cy_tile, self.cfg.TMS_ZOOM_LEVEL)
```

Optionally blend PF and EKF centers:

```python
center = weighted_center(pf_center, ekf_center, w=confidence_from_n_eff)
```

### Validation test

Create a controlled test where EKF prior is offset by 300 m but PF center is correct. The first-pass tile list should move around the PF center, not the EKF center.

> **[Claude Code review]:** Confirmed in code — `temporal_searcher.py:299` sets `center_lat, center_lon = imu_data["lat"], imu_data["lon"]` and `region["center"]` (the PF's weighted mean position) is never used. The radius from `region["radius_tiles"]` IS used but the center is EKF.
>
> **Severity disagreement: I rate this High, not Critical.** Here is why. The PF receives visual updates from the same homography measurements that update the EKF. In normal (non-drifted) operation they track each other closely. The divergence only matters when: (a) the EKF has drifted but the PF particles have stayed near the true position, or (b) after a relocalization reset. In case (b), Phase R sets `searcher.frame_count = 0` which triggers a cold-start PF reset on the next frame around the corrected EKF position anyway. In case (a), the PF is fed EKF-filtered tile-center measurements (`pf_update_source = "tile_center"`) which are gated by `max_pf_innovation_m` relative to the EKF position. So during an EKF drift event, the PF is also being fed wrong measurements relative to the EKF, and is not guaranteed to stay correct either. This is a real design problem, but calling it Critical without experimental evidence that PF and EKF actually diverge usefully seems too strong.
>
> **Additional finding:** The trace logging has a related bug. `_trace_data['pf_center']` is set to `(center_lat, center_lon)` which is `imu_data["lat"], imu_data["lon"]` — the EKF position. Every `trace.json` has a `pf_center` field that is actually the EKF position, not the particle filter weighted mean. The real PF center is `region["center"]` from `get_search_region()`. This mislabels every trace file and misleads anyone analyzing the PF behavior offline.

---

## CP-06 — The old `position_estimator.py` TMS Y conversion is inconsistent with the newer visual-measurement conversion

**Severity:** Critical  
**Files:** `position_estimator.py`, `visual_measurement.py`, `meta_tile_builder.py`

### Evidence

`MetaTileBuilder.build_meta_tile()` places the northernmost tile at row 0 using:

```python
row = y_max - ty
```

`visual_measurement.py` converts meta-tile pixels using:

```python
tile_y_frac = (y_max + 1) - px_y / tile_px
```

That is correct for a north-up canvas.

But `position_estimator.py` uses:

```python
tile_y_frac = y_min + ref_px_y / tile_px
```

and for single tiles:

```python
tile_y_frac = tile_y + ref_px_y / tile_px
```

This is the opposite vertical convention.

### Why this matters

Cold-start logic and any legacy code using `position_estimator.py` can produce vertically flipped latitude estimates inside a tile/meta-tile. At zoom 16, this can be hundreds of meters.

### Suggested fix

Delete duplicate pixel-to-lat/lon logic from `position_estimator.py` and reuse one canonical function:

```python
def metatile_pixel_to_latlon(px_x, px_y, tiles, tile_px, zoom):
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    x_min = min(xs)
    y_max = max(ys)
    tile_x_frac = x_min + px_x / tile_px
    tile_y_frac = (y_max + 1) - px_y / tile_px
    return tile_to_latlon(tile_x_frac, tile_y_frac, zoom)
```

For a single tile:

```python
tile_y_frac = tile_y + 1 - ref_px_y / tile_px
```

### Validation test

For one known tile:

- pixel `(0, 0)` must map to the **north-west** corner,
- pixel `(512, 512)` must map to the **south-east** corner,
- pixel `(256, 256)` must map to the tile center.

> **[Claude Code review]:** Confirmed. Verified in source: `position_estimator.py:pixel_to_latlon_in_metatile()` uses `y_min + ref_px_y / tile_px` (wrong — Y increases into the tile = goes north, but pixels increase downward = south). `pixel_to_latlon_single_tile()` uses `tile_y + ref_px_y / tile_px` (same bug). `visual_measurement.py:px_to_latlon()` uses `(y_max + 1) - px_y / tile_px` (correct — Y decreases as pixels go down).
>
> **Severity caveat:** the hot path no longer calls `position_estimator.py` functions. `_process_frame_N` uses `visual_measurement.py` directly. `_process_frame_0` also uses `extract_visual_measurements()` for the quality-gate path. The broken functions in `position_estimator.py` are only reached in the cold-start fallback path when gate_pass=False but score≥100 (`search_result["position"]` from BestFirstSearcher). Still a real bug that should be fixed, but the "hundreds of meters" error only affects the PF initialization in the medium-confidence cold-start case, not the EKF update path.

---

## CP-07 — Cold-start still uses the old single-tile estimator

**Severity:** High  
**Files:** `best_first_search.py`, `position_estimator.py`, `temporal_searcher.py`

### Evidence

`BestFirstSearcher.search()` uses:

```python
pixel_to_latlon_single_tile(ref_x, ref_y, best_tx, best_ty, ...)
```

That function currently has the likely TMS-Y inversion problem described in CP-06.

### Why this matters

If frame 0 passes the quality gate, the particle filter can be initialized from a wrong visual position. That can poison the entire temporal run.

### Suggested fix

Cold-start should use the same visual-measurement stack as temporal tracking:

- `compute_dual_homography()`
- `extract_visual_measurements()`
- same cascade
- same canonical pixel-to-lat/lon conversion

### Validation test

Pick a frame with a known tile and manually verify that projected center lands in the expected geographical quadrant.

> **[Claude Code review]:** Partially confirmed, but severity is lower than stated. The current `_process_frame_0` already uses `compute_dual_homography()` and `extract_visual_measurements()` for the visual quality gate path (CShape + inliers check). When gate_pass=True, `homo_position` comes from `extract_visual_measurements()` — the correct conversion. The PF is then initialized at this correct position.
>
> The bug only activates when gate_pass=False but score≥100, where the fallback `position = search_result["position"]` from BestFirstSearcher is used. That position comes from `pixel_to_latlon_single_tile()` in `position_estimator.py` — the wrong conversion. This is used only for PF initialization (not EKF update) in the medium-confidence cold-start case.
>
> **I would rate this Medium, not High**, because: (1) the EKF is never updated from the wrong position, (2) it only affects PF initialization in the medium-confidence fallback, (3) the PF will typically correct itself within a few frames once visual updates arrive. Still worth fixing — the suggestion to always use `extract_visual_measurements()` is correct and consolidates the code.

---

## CP-08 — Semantic confirmation is not actually a confirmation gate

**Severity:** High  
**File:** `temporal_searcher.py`, `semantic_confirmer.py`, `run_pipeline.py`

### Evidence

`TemporalSearcher._process_frame_N()` computes:

```python
confirm_result = self.semantic_confirmer.confirm(...)
```

But it does not use this result to reject or accept the visual measurement. It only returns:

```python
"semantic_confidence": confirm_result["confidence"]
```

Then `run_pipeline.py` uses `semantic_conf` only to scale `R`:

```python
r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
```

### Why this matters

The code calls it semantic confirmation, but semantically weak frames are not rejected. They are only given different measurement covariance. That is not confirmation; it is soft weighting.

### Suggested fix

Rename the concept to `semantic_similarity`, or use it as an explicit gate:

```python
semantic_ok = sem_conf >= SEMANTIC_CONFIRMATION_MIN
```

Then log:

- `semantic_similarity`
- `semantic_gate_pass`
- `geometry_gate_pass`
- `innovation_gate_pass`
- `ekf_update_applied`

### Validation test

Run frames with deliberately wrong reference tiles. Semantic similarity should drop and prevent EKF update.

> **[Claude Code review]:** Confirmed as a naming/concept inconsistency. However, I partially disagree with the hard gate suggestion.
>
> **On the naming:** Correct. Calling it `semantic_confirmation` when it only adjusts R is misleading. Rename to `semantic_similarity` or `semantic_weight`.
>
> **On the hard gate:** Soft R-scaling is actually more Bayesian than a hard gate — it continuously trades off visual uncertainty against semantic evidence. A hard gate loses information: a frame with semantic_conf=0.69 (just below threshold) would be fully rejected even though it has useful localization information. The R-scaling already does the right thing: low semantic confidence inflates R, reducing EKF update weight proportionally. The risk of a hard gate is over-rejection when the semantic model is uncertain about terrain type or when prediction tiles are slightly misaligned.
>
> **The real problem** is that the scaling formula `max(0.5, 2.0 - 1.5 * sem_conf)` produces multipliers between 0.5 and 2.0, which is relatively mild. A semantic_conf of 0.5 gives R × 1.25 — barely a 25% increase. If semantic evidence is genuinely wrong (conf=0.1), R × 1.85 — still allows the update with only slightly larger uncertainty. The formula could be made steeper, or log-scaled, before resorting to a hard gate.
>
> **The renaming and separate logging column** are the right priority: log `semantic_similarity` and `semantic_gate_pass` (with a configurable threshold) so you can observe the behavior and decide empirically whether a hard gate helps.

---

## CP-09 — PF innovation rejection reason can be overwritten by runtime innovation logic

**Severity:** Medium  
**Files:** `temporal_searcher.py`, `run_pipeline.py`

### Evidence

`TemporalSearcher._process_frame_N()` can set:

```python
visual_rejected_reason = "pf_innovation_too_large"
```

But `run_pipeline.py` later initializes its own:

```python
visual_rejected_reason = ""
```

and then writes that back into the result.

### Why this matters

A frame can be rejected by the PF logic but the final CSV can hide that reason. That destroys diagnostic value.

### Suggested fix

Use separate fields:

```python
pf_rejected_reason
runtime_rejected_reason
ekf_rejected_reason
```

Or combine them:

```python
reasons = []
if result.get("visual_rejected_reason"):
    reasons.append(result["visual_rejected_reason"])
if innovation_too_large:
    reasons.append("ekf_innovation_too_large")
```

### Validation test

Force one frame where PF innovation fails but EKF innovation passes. The CSV must show both facts.

> **[Claude Code review]:** Confirmed, and the affected scope is wider than described. In `_process_one_frame`, `visual_rejected_reason = ""` is initialized locally, then only set to `"innovation_too_large"` if the EKF gate fires. `result["visual_rejected_reason"] = visual_rejected_reason` overwrites the PF's reason at the end. The following cases all produce an empty `visual_rejected_reason` in results.csv even though the frame was rejected:
>
> 1. PF innovation too large (`pf_innovation_too_large` from temporal_searcher) — overwritten by `""`.
> 2. Visual quality gate fail (CShape < threshold or inliers < threshold) — `temporal_searcher` returns `gate_pass=False` but `visual_rejected_reason=""` because the reason field is only set for PF innovation, not for quality gate failure. The EKF gate then skips (`gate_pass=False` → EKF branch not reached), so `visual_rejected_reason` stays `""`.
>
> The result: frames rejected by CShape/inliers and frames rejected by PF innovation both silently appear as `gate_pass=0, visual_rejected_reason=""` in the CSV. Only EKF innovation rejections produce a non-empty reason. This affects all three gate layers, not just the one described.

---

## CP-10 — `gate_pass` means too many different things

**Severity:** High  
**Files:** `temporal_searcher.py`, `run_pipeline.py`, `results.csv`

### Evidence

There are at least three different gate concepts:

1. Visual quality gate: CShape + inliers + homography position.
2. PF innovation gate: homography plausible relative to PF/EKF prior.
3. Runtime EKF innovation gate: corrected homography plausible relative to EKF state.

But the CSV has one main `gate_pass` field.

### Why this matters

A single boolean cannot explain the decision chain. This is why debugging becomes confusing.

### Suggested fix

Replace the single gate with explicit columns:

| Column | Meaning |
|---|---|
| `geometry_quality_pass` | CShape/inliers/homography valid. |
| `pf_innovation_pass` | Visual measurement plausible for PF update. |
| `semantic_gate_pass` | Semantic similarity acceptable. |
| `ekf_innovation_pass` | Visual measurement accepted by EKF gate. |
| `ekf_update_applied` | EKF was actually updated. |

Keep `gate_pass` only as a final alias for `ekf_update_applied` if needed.

### Validation test

Frames 84, 91, 93, 95, 118, and 122 from the uploaded run should clearly show which gate rejected them.

> **[Claude Code review]:** Partially implemented after Phase R. The current RESULT_COLUMNS now includes `visual_quality_pass` (geometry quality before innovation gate), `ekf_update_applied` (EKF was actually updated), `relocalization_candidate`, and `relocalization_applied`. What remains missing: `pf_innovation_pass` (whether PF inner gate passed), `semantic_gate_pass` (would need a threshold defined). The `visual_quality_pass` column is equivalent to `geometry_quality_pass` from the suggested table. The gap is specifically the PF innovation gate visibility — as described in CP-09, PF rejections are currently invisible in the CSV.

---

## CP-11 — The visual quality gate accepts high-error frames

**Severity:** Critical  
**Files:** `results.csv`, `run_pipeline.py`, `temporal_searcher.py`

### Evidence

Uploaded run examples:

- Frame 113: 194.8 m error, CShape 0.591, 618 inliers, gate PASS.
- Frame 114: 189.6 m error, CShape 0.467, 567 inliers, gate PASS.
- Frame 115: 179.3 m error, CShape 0.412, 406 inliers, gate PASS.

### Why this matters

Inliers alone are not proof of correct localization. A wrong local patch with similar visual texture can still produce hundreds of matches.

### Suggested fix

Add stronger visual measurement quality metrics:

| Additional check | Why |
|---|---|
| Inlier spatial coverage | Prevent matches concentrated in one local patch. |
| Median reprojection error | Reject geometrically sloppy fits. |
| Homography condition / projective distortion | Reject unstable mappings. |
| Distance between projected center and inlier centroid | Detect suspicious extrapolation. |
| Semantic similarity gate | Reject wrong terrain context. |
| Temporal consistency of consecutive visual positions | Detect jumps even if single-frame quality looks good. |

### Validation test

Create a table for all accepted frames:

```text
error_m, CShape, inliers, reproj_median, spatial_coverage, semantic_conf
```

Then tune thresholds on validation flights, not by guessing.

> **[Claude Code review]:** Confirmed. This is the hardest unsolved problem. Note that the `compute_dual_homography()` function DOES already compute `reproj_median` for both DLT and MAGSAC branches and stores them in the result dict — they are just not used anywhere downstream. The reprojection information is available without any additional computation; it only needs to be plumbed through to the gate and logged. Also: `inlier_ratio` (inliers / total matches) is not computed anywhere, only raw inlier count. A high absolute inlier count with a low ratio (e.g., 618 inliers out of 2048 = 30%) is less trustworthy than 150 inliers out of 200 (75%). Both metrics are needed.

---

## CP-12 — Runtime innovation gate is too permissive for high-speed flight

**Severity:** High  
**File:** `run_pipeline.py`

### Evidence

The innovation threshold is computed as:

```python
max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * dt_gate + 50.0)
```

In the uploaded run, accepted frame 113 had:

- visual innovation: 181.1 m
- max innovation: 238.9 m
- final error: 194.8 m

### Why this matters

The gate can accept very large jumps if the aircraft is fast or the EKF covariance is loose. That may keep the filter from rejecting wrong but plausible visual measurements.

### Suggested fix

Use normalized innovation squared (NIS) gating instead of a hand-built distance threshold:

```python
innovation = z - h
S = H @ P @ H.T + R
nis = innovation.T @ inv(S) @ innovation
accept = nis < chi2.ppf(0.99, df=2)
```

This naturally accounts for EKF covariance and measurement covariance.

### Validation test

Log `nis_position` for every visual update and compare rejected/accepted distributions.

> **[Claude Code review]:** Confirmed in principle. However, the NIS implementation is more complex than it appears here because: (a) it requires the current H and R matrices at gate time (not just scalars), and (b) the EKF position states are error-states (deltas), not absolute positions — so the innovation vector needs to be expressed in error-state space. The formula as written assumes a direct observation model.
>
> **Simpler first step:** the 150m floor is the primary culprit. Frame 113 had pos_sigma=~80m, so `3×80 + 67×2.5 + 50 = 457m` max — very wide. But the actual innovation was only 181m, well inside 457m. The 150m floor had nothing to do with F113. The issue is that at the actual drone speed (~67 m/s) with actual dt (~2.5s), the formula produces thresholds over 400m, which is permissive enough to accept 200m errors. Reducing the `vel` coefficient from 1.0 to 0.5 or reducing the `3σ` factor to `2σ` would tighten the gate without requiring full NIS. That should be validated first. NIS is the theoretically correct approach but requires more refactoring.

---

## CP-13 — Look-ahead correction is hard-coded and outside `config.py`

**Severity:** High  
**File:** `run_pipeline.py`

### Evidence

```python
LOOKAHEAD_M = 110.0
```

The correction is applied directly in `run_pipeline.py`.

### Why this matters

A hard-coded 110 m offset might work for one camera view, one aircraft speed, one altitude range, and one MSFS setup. It is not a general camera model. If altitude, FOV, pitch, camera mount angle, or screen capture geometry changes, the correction becomes wrong.

### Suggested fix

Move it to `config.py` immediately:

```python
CAMERA_LOOKAHEAD_M = 110.0
```

Then replace it with a calibrated model later:

```python
lookahead_m = f(altitude_m, pitch_rad, camera_mount_angle, fov_y)
```

### Validation test

Use uploaded analysis logic to estimate residual offset vs heading for multiple flights. Look-ahead should be fitted from data, not guessed.

> **[Claude Code review]:** Confirmed. Also: `R_HIGH = 30.0 ** 2`, `R_MED = 60.0 ** 2`, `R_COLD_START = 10000.0`, `TURN_ROLL_THRESHOLD_RAD = 0.35`, and `TURN_R_MULTIPLIER = 2.0` are all module-level constants in `run_pipeline.py` (lines 55–59) rather than in `config.py`. These control EKF measurement noise and turn handling — they are tuning parameters that belong in config alongside the existing `VISUAL_POSITION_NOISE_M`, `POSITION_PROCESS_NOISE_M`, and `INITIAL_POSITION_VARIANCE_M` constants that are already in `config.py`. Moving all of them to `config.py` would let `run_meta.json` capture the full tuning state with a config snapshot.

---

## CP-14 — Nadir correction uses an approximate FOV and ignores actual altitude/camera calibration

**Severity:** High  
**File:** `visual_measurement.py`

### Evidence

`extract_visual_measurements()` accepts `altitude_m`, but the actual nadir shift uses:

```python
f_px_approx = query_w / (2 * math.tan(math.radians(35)))
nadir_x = cx - f_px_approx * math.tan(roll_rad)
nadir_y = cy - f_px_approx * math.tan(pitch_rad)
```

The `altitude_m` argument is not used in the computation.

### Why this matters

This is not a physically grounded camera projection. It assumes a 70° field of view and no calibrated intrinsic matrix. Worse, `_build_cascade()` always puts `nadir_corrected` first, so this approximate method dominates measurement selection.

### Suggested fix

Short-term:

- rename `nadir_corrected` to `attitude_shifted_center_approx`, or
- demote it behind `trimmed_centroid` unless validated.

Better fix:

- use calibrated camera intrinsics `K`,
- use camera-to-body extrinsics,
- use altitude above ground,
- ray-cast the camera optical axis to the ground plane.

### Validation test

Compare these measurement methods against GPS ground truth:

- projected center
- trimmed centroid
- inlier centroid
- weighted centroid
- current nadir-corrected

Keep only the method that wins consistently by error statistics.

> **[Claude Code review]:** Confirmed. Two additional issues found in the code:
>
> 1. The formula `f_px_approx = query_w / (2 * math.tan(math.radians(35)))` assumes `query_w` corresponds to a 70° FOV. But `query_w` is the resized image width (capped at `MAX_ROTATED_DIMENSION = 1280` by `_resize_rotated()`), not the original 1920px. If the original image is 1920px wide with a 70° FOV and is resized to 1280px, the effective focal length at the resized resolution changes proportionally. The formula is computing `f_px` for the resized width as if the FOV is still 70°, but a smaller image from the same lens has the same focal length in pixels only if the image is cropped, not resized. This introduces a systematic error in the nadir shift estimation for every resized frame.
>
> 2. The `_build_cascade()` comment says "nadir_corrected is always first — it shifts the projected nadir ground-point for both pitch and roll, and is MOST valuable when the aircraft is banking." This is a deliberate design choice, but it was made without validation data. An aircraft banking at 20° has `roll=0.35 rad`, `tan(roll)=0.36`, nadir shift = `0.36 × f_px`. At 1280px width with the formula: `f_px ≈ 1280 / (2 × 0.70) ≈ 914px`, shift ≈ 329px. That is a very large shift applied on every banking frame, derived from an unvalidated FOV. If the FOV assumption is wrong by 10°, the shift is wrong by ~15%.

---

## CP-15 — Homography winner selection ignores reprojection error

**Severity:** High  
**File:** `visual_measurement.py`

### Evidence

`compute_dual_homography()` computes reprojection errors for DLT and MAGSAC, but `_select_homography_winner()` scores only:

```python
score = inliers * CShape * convexity_bonus
```

### Why this matters

A homography can have many inliers and a decent projected shape but still be geometrically sloppy. Reprojection error should be part of the decision.

### Suggested fix

Use a composite score like:

```python
score = (
    inlier_ratio
    * CShape
    * spatial_coverage
    / (1.0 + median_reproj_error)
)
```

Also apply hard rejection:

```python
if median_reproj_error > REPROJ_MEDIAN_MAX:
    reject
```

### Validation test

Log winner branch, DLT/MAGSAC scores, reprojection medians, and final error. Verify that the winner correlates with lower GPS error.

> **[Claude Code review]:** Confirmed. Additional note: the current score uses absolute inlier count `n`, not inlier ratio `n / total_correspondences`. A noisy match with 2000 correspondences and 600 inliers (30% ratio, CShape=0.4) scores `600 × 0.4 = 240`. A clean match with 200 correspondences and 180 inliers (90% ratio, CShape=0.8) scores `180 × 0.8 = 144`. The noisy match wins despite being geometrically less consistent. Replacing `n` with `inlier_ratio` (or using `n × inlier_ratio`) would reward high-precision matches over high-volume noisy matches.

---

## CP-16 — `position_estimator.py` is partly legacy but still callable

**Severity:** Medium  
**File:** `position_estimator.py`

### Evidence

The temporal path mostly uses `visual_measurement.py`, but `best_first_search.py` still imports and uses `estimate_homography()`, `query_center_in_reference()`, and `pixel_to_latlon_single_tile()` from `position_estimator.py`.

### Why this matters

Legacy helpers are dangerous when they still sit on an active path. A reader will assume they are equivalent to the newer visual measurement code. They are not.

### Suggested fix

Choose one:

1. Make `position_estimator.py` a thin compatibility wrapper around `visual_measurement.py`, or
2. Delete it after moving any still-needed functions into the canonical visual module.

### Validation test

Search imports. Only one module should own homography-to-GPS conversion.

> **[Claude Code review]:** Confirmed. The reviewer's severity of Medium is appropriate given the current state. The only hot path still using `position_estimator.py` is `BestFirstSearcher.search()` for the fallback cold-start position (score≥100, gate_pass=False). The fix is straightforward: in `_process_frame_0`, replace the BFS fallback `position` with the output of `extract_visual_measurements()` using the best single tile, keeping the same cascade logic already used in the quality gate path.

---

## CP-17 — `results.csv` schema documentation is stale

**Severity:** Low  
**File:** `run_pipeline.py`

### Evidence

The module docstring says `Columns in results.csv (21)`, but `RESULT_COLUMNS` currently contains 41 columns.

### Why this matters

Small, but thesis readers and future debugging scripts will be misled.

### Suggested fix

Generate the docstring table from `RESULT_COLUMNS`, or remove the hard-coded count.

### Validation test

Add a unit test:

```python
assert len(RESULT_COLUMNS) == len(pd.read_csv(results_path).columns)
```

> **[Claude Code review]:** Confirmed. The count in the docstring (21) is clearly stale — `RESULT_COLUMNS` has 41 entries after Phase R additions. The simplest fix is to remove the count from the docstring and let the actual list speak for itself.

---

## CP-18 — Config comment says Copenhagen while paths point to Odense

**Severity:** Low  
**File:** `config.py`

### Evidence

Comment:

```python
# Reference TMS tileset (Copenhagen, Denmark - zoom 16)
```

Actual paths:

```python
REFERENCE_TILES_DIR = ... / "REFERENCE_MAP_ODENSE" / "aerial"
REFERENCE_PRED_DIR = ... / "REFERENCE_MAP_ODENSE" / "prediction"
```

### Why this matters

It is a documentation bug, but it creates real confusion when writing the thesis.

### Suggested fix

Change the comment to:

```python
# Reference TMS tileset (Odense, Denmark - zoom 16)
```

Or generalize:

```python
# Active reference TMS tileset selected by REFERENCE_MAP_* paths.
```

> **[Claude Code review]:** Confirmed. One-line fix. The tile bounds (`TILE_X_MIN/MAX`, `TILE_Y_MIN/MAX`) and the latitude used for `TILE_SIZE_METERS` computation (`_LAT_RAD = math.radians(55.6)`) are all consistent with Odense (~55.4°N). The comment is the only thing that says Copenhagen.

---

## CP-19 — `DEVICE = "cuda"` is not safe for every module

**Severity:** Medium  
**Files:** `config.py`, `semantic_model.py`, `geometric_matcher.py`

### Evidence

`config.py` sets:

```python
DEVICE = "cuda"
```

`SemanticModel` internally falls back to CPU if CUDA is unavailable, but `SuperPointLightGlueMatcher` uses the given device directly:

```python
.to(device)
```

### Why this matters

On a CPU-only machine, semantic inference may survive but the feature matcher can crash.

### Suggested fix

Centralize device resolution:

```python
def resolve_device(preferred="cuda"):
    return "cuda" if preferred == "cuda" and torch.cuda.is_available() else "cpu"
```

Use it for both semantic model and matcher.

### Validation test

Run import and model initialization on CPU-only environment or with `CUDA_VISIBLE_DEVICES=""`.

---

## CP-20 — Feature store is opened but never explicitly closed

**Severity:** Medium  
**File:** `run_pipeline.py`

### Evidence

`_init_models()` does:

```python
feature_store = FeatureStoreLoader(...)
feature_store.open()
```

No corresponding close is visible in `run_file_mode()` or `run_simconnect_mode()`.

### Why this matters

For short runs this probably survives. For repeated runs, tests, or notebook restarts, HDF5 handles can stay open and cause file-locking or resource issues.

### Suggested fix

Add:

```python
try:
    ...
finally:
    if feature_store is not None:
        feature_store.close()
```

Or make `FeatureStoreLoader` a context manager.

### Validation test

Run 10 short runs in a loop and ensure no file-handle leak or HDF5 lock error occurs.

---

## CP-21 — JSON trace writer can emit non-standard `NaN`

**Severity:** Medium  
**File:** `run_pipeline.py`

### Evidence

`_json_default()` handles NumPy scalar conversion, but Python's `json.dumps()` does not call `default` for normal Python `float("nan")`. By default, `json.dumps()` allows `NaN`, which is not strict JSON.

### Why this matters

Some JSON parsers reject `NaN`. This can break downstream tools that expect standard JSON.

### Suggested fix

Implement a recursive sanitizer and use strict JSON:

```python
json.dumps(clean_nan(obj), allow_nan=False)
```

### Validation test

Run:

```python
json.loads(trace_path.read_text())
```

using a strict parser or validate with a JSON linter.

> **[Claude Code review]:** Confirmed. The `_json_default` handler correctly intercepts `np.floating` NaN and returns `None`, but it is never called for plain Python `float("nan")` — the json module's standard encoder handles those itself, and with `allow_nan=True` (the default) writes them as the bare token `NaN`. If any pipeline variable is a Python float NaN (e.g., from `float("nan")` or arithmetic returning nan), it silently writes non-standard JSON. Using `json.dumps(..., allow_nan=False)` would immediately surface any such values as `ValueError` during testing.

---

## CP-22 — `FileSource` timestamp alignment can silently drop frames

**Severity:** Medium  
**File:** `simconnect_adapter.py`

### Evidence

Frame files are mapped by:

```python
frame_map[round(float(ts_str), 3)] = fp
```

Rows are matched by:

```python
ts_rounded = round(row["timestamp"], 3)
if ts_rounded in frame_map:
    yield ...
```

### Why this matters

Rounding to 3 decimals is brittle. Two frames can collide after rounding, and small timestamp drift can silently drop frames.

### Suggested fix

Use nearest-neighbor matching with tolerance:

```python
nearest = min(frame_times, key=lambda t: abs(t - row_ts))
if abs(nearest - row_ts) <= 0.02:
    yield ...
```

Also print alignment stats:

```text
frames found, rows matched, rows skipped, max timestamp error
```

### Validation test

Create artificial timestamps offset by ±0.004 s and verify frames still align.

---

## CP-23 — `pressure_altitude` documentation contradicts itself

**Severity:** Low  
**File:** `simconnect_adapter.py`

### Evidence

Top unit contract says:

```text
pressure_altitude — metres
```

But `slow_cache` comment says:

```python
"pressure_altitude": None,   # feet
```

Later line says it is metres again.

### Why this matters

Altitude unit bugs are brutal. This comment conflict invites future mistakes.

### Suggested fix

Use one statement everywhere:

```python
pressure_altitude — metres, as returned by Python SimConnect in this setup
```

### Validation test

Add a unit test checking barometric fallback does not multiply `pressure_altitude` by 0.3048.

> **[Claude Code review]:** Confirmed. Looking at `simconnect_adapter.py:251`: the `slow_cache` initialization block has `"pressure_altitude": None,   # feet` — but the later assignment at line 294 has `slow_cache["pressure_altitude"] = aq.get("PRESSURE_ALTITUDE")  # metres (Python SimConnect returns SI)`. The unit contract at the top of the file also says metres. The `# feet` in the slow_cache initialization is wrong and should be removed or corrected to `# metres`.

---

## CP-24 — The semantic class palette is duplicated

**Severity:** Medium  
**Files:** `config.py`, `semantic_model.py`, `semantic_tile_scorer.py`

### Evidence

Class names and color maps appear separately in:

- `config.py`
- `semantic_model.py`
- `semantic_tile_scorer.py`

### Why this matters

If one class color changes, parts of the pipeline can silently disagree. That is especially dangerous because semantic maps are used for both prefiltering and confidence scoring.

### Suggested fix

Create one module:

```python
src/semantic_palette.py
```

containing:

```python
SEMANTIC_CLASSES
COLOR_MAP
COLOR_TO_CLASS
NUM_CLASSES
```

Import it everywhere.

### Validation test

Add:

```python
assert semantic_model.COLOR_MAP == config.COLOR_MAP
assert semantic_tile_scorer.COLOR_TO_CLASS == inverse(config.COLOR_MAP)
```

---

## CP-25 — Unknown semantic colors are silently treated as water

**Severity:** High  
**File:** `semantic_tile_scorer.py`

### Evidence

`_rgb_to_class_mask()` initializes:

```python
mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
```

Pixels that match no palette color remain class `0`, i.e. waterbodies.

### Why this matters

If prediction tiles contain anti-aliased colors, compression artifacts, or palette drift, unknown pixels become water. That can completely distort semantic histograms.

This is especially risky because earlier dataset work already had color-encoding issues.

### Suggested fix

Use `255` for unknown:

```python
mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
```

Then compute histograms ignoring `255`, and log `unknown_pixel_ratio`.

### Validation test

Feed a prediction tile with one wrong RGB value. It should report unknown pixels, not water.

---

## CP-26 — The fallback semantic confirmation path preprocesses meta-tiles like query frames

**Severity:** Medium  
**File:** `semantic_confirmer.py`

### Evidence

If no prediction meta-tile exists, `SemanticConfirmer.confirm()` does:

```python
processed = preprocess_query_frame(meta_tile, resize_w=..., resize_h=..., target_size=512)
```

### Why this matters

A meta-tile is not a 1920×1079 camera frame. It can be 512×512, 1024×512, 1536×1536, etc. Query-frame padding/resizing is not the right preprocessing for reference meta-tiles.

### Suggested fix

Use a separate preprocessing function:

```python
preprocess_reference_image_for_semantic_model(meta_tile)
```

Or require precomputed prediction tiles and remove this fallback.

### Validation test

Create a 1536×1536 meta-tile and verify the semantic mask is not distorted by camera-frame padding logic.

---

## CP-27 — Current latency is too high for real-time navigation at the tested speed

**Severity:** Critical for deployment, Medium for thesis prototype  
**Files:** `results.csv`, `run_pipeline.py`, `temporal_searcher.py`, `meta_tile_builder.py`

### Evidence

Uploaded `results.csv`:

- mean inference latency: 2416 ms
- median inference latency: 2434 ms
- P90 latency: 2712 ms
- max latency: 3305 ms

At 60-70 m/s, the aircraft travels roughly 145-190 m during one inference cycle.

### Why this matters

Even a perfect estimate can be stale by the time it is emitted. This directly limits live use.

### Suggested fix

Short-term:

1. Timestamp every visual measurement at frame capture time.
2. Propagate the visual measurement forward to output time using EKF velocity.
3. Log `state_at_capture`, `state_at_output`, and `latency_compensated_position`.

Performance fixes:

- use cached prediction tiles,
- cache loaded aerial tiles,
- reduce candidate count adaptively,
- avoid repeated full-frame resize/copy,
- run semantic inference less frequently or asynchronously,
- use TensorRT/ONNX for segmentation if targeting embedded runtime,
- benchmark feature-store hit rate.

### Validation test

Log:

```text
capture_time, visual_estimate_time, output_time, aircraft_displacement_during_inference
```

Then evaluate both uncompensated and latency-compensated outputs.

> **[Claude Code review]:** Confirmed. The latency compensation suggestion (propagate to output time using EKF velocity) is well worth doing as an immediate fix. The EKF velocity is always available (vel_n, vel_e from `ekf.get_state()`), and the latency from frame capture to position output is already tracked by `frame_capture_ts` and `gps_estimate_ts`. This is a 5-line correction after each frame: shift `final_lat/lon` forward by `(gps_estimate_ts - frame_capture_ts) * vel_n/vel_e`. This doesn't improve accuracy but eliminates the systematic temporal bias from slow inference. Note: `inference_ms` is already logged in live mode (`time.perf_counter() - frame_capture_ts`) so the data is already available.

---

## CP-28 — Meta-tile and prediction tile loading is not cached

**Severity:** Medium  
**Files:** `tile_utils.py`, `meta_tile_builder.py`

### Evidence

`TileLoader.load_aerial()` and `TileLoader.load_prediction()` read from disk every call.

`build_prediction_meta_tile()` first checks whether any prediction tile exists, then loads tiles again during stitching.

### Why this matters

Disk I/O is not the biggest latency source compared to feature matching, but repeated tile loads are unnecessary overhead in a real-time pipeline.

### Suggested fix

Add an LRU cache:

```python
@lru_cache(maxsize=512)
def load_aerial_cached(tile_x, tile_y):
    ...
```

Or implement cache inside `TileLoader`.

### Validation test

Log cache hit rate and compare average `meta_tile_ms` before/after.

---

## CP-29 — `MetaTileBuilder.run()` docstring says "save" but saving is debug-only

**Severity:** Low  
**File:** `meta_tile_builder.py`

### Evidence

The docstring says:

```text
first pass → second pass → build → save → verify
```

But actual code only saves when:

```python
DEBUG_SAVE_METATILES == True
```

### Why this matters

Not a runtime bug, but the documentation is stale.

### Suggested fix

Change wording:

```text
first pass → second pass → build → optional debug save → verify
```

---

## CP-30 — Run metadata is too thin

**Severity:** Medium  
**File:** `run_pipeline.py`

### Evidence

`run_meta.json` contains run ID, source, frame count, gate count, elapsed time, FPS, and input paths.

It does not store:

- config snapshot,
- git commit/hash,
- map name,
- model checkpoint path/hash,
- feature store path/hash,
- camera/lookahead constants,
- active save flags,
- tile bounds,
- CUDA/device info.

### Why this matters

You cannot reproduce a run precisely from `run_meta.json` alone.

### Suggested fix

Write:

```json
{
  "config": {...},
  "model_checkpoint": "...",
  "reference_map": "...",
  "feature_store": "...",
  "git_commit": "...",
  "device": "cuda",
  "constants": {...}
}
```

### Validation test

A future analysis script should be able to regenerate labels and understand the run without opening `config.py`.

> **[Claude Code review]:** Confirmed and directly relevant to thesis reproducibility. The minimal fix: add `subprocess.run(["git", "rev-parse", "HEAD"])` to get the commit hash, and dump a filtered `vars(config)` (excluding Path objects, converting them to strings) as the config snapshot. The R_HIGH/R_MED/LOOKAHEAD_M inline constants in run_pipeline.py (see CP-13) would NOT be captured by a config dump — another reason to move them into config.py first.

---

## CP-31 — WMM coefficient path is brittle

**Severity:** Medium  
**File:** `wmm_declination.py`

### Evidence

The coefficient path is built as:

```python
Path(__file__).resolve().parents[2] / "WMM2025COF" / ...
```

### Why this matters

It works only if the folder layout stays exactly the same. Moving `src/`, packaging the project, or running from a different installed layout can break magnetic declination lookup.

### Suggested fix

Move WMM path to config:

```python
WMM2025_COF_PATH = ALL_IN_ONE_ROOT / "WMM2025COF" / "WMM2025COF" / "WMM2025.COF"
```

Then pass it or import from config.

### Validation test

Run from project root, from `runtime/`, and from an installed package layout.

---

## CP-32 — `ErrorStateEKF` is doing too many jobs in one file

**Severity:** Medium  
**File:** `ekf_ins.py`

### Evidence

`ekf_ins.py` contains:

- quaternion math,
- barometric altitude,
- EKF class,
- online `step_ekf()` wrapper,
- batch CSV processing,
- output DataFrame construction,
- GPS leakage check.

### Why this matters

The math-heavy EKF code is mixed with I/O and analysis helpers. That makes it harder to safely review or test.

### Suggested fix

Split into:

| New file | Content |
|---|---|
| `ekf/quaternion.py` | quaternion helpers |
| `ekf/filter.py` | `ErrorStateEKF` |
| `ekf/msfs_units.py` | MSFS-to-NED conversion |
| `ekf/step.py` | `step_ekf()` |
| `ekf/batch.py` | `preprocess_imu_csv()` |

### Validation test

All existing EKF tests must pass without changing numerical outputs.

---

## CP-33 — EKF covariance update should be numerically guarded

**Severity:** Medium  
**File:** `ekf_ins.py`

### Evidence

The EKF uses covariance updates like:

```python
self.P = self.P - K @ S @ K.T
```

This is algebraically valid under exact arithmetic, but it is less numerically robust than Joseph-form covariance update.

### Why this matters

Long runs can accumulate numerical asymmetry or small negative variances.

### Suggested fix

Use Joseph form for measurement updates:

```python
I = np.eye(self.P.shape[0])
self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
self.P = 0.5 * (self.P + self.P.T)
```

### Validation test

After every EKF update:

```python
assert np.allclose(P, P.T, atol=1e-9)
assert np.min(np.linalg.eigvalsh(P)) > -1e-9
```

> **[Claude Code review]:** Confirmed via grep: `self.P = self.P - K @ S @ K.T` appears at 4 locations in `ekf_ins.py`. This is algebraically equivalent to `(I - KH)P` (the standard simplified form) when P is symmetric — it is NOT the Joseph form `(I - KH)P(I - KH)^T + KRK^T`. The current form is correct under exact arithmetic but loses symmetry and positive-definiteness under floating-point rounding over long runs.
>
> **Additional finding:** there is no symmetrization step (`P = 0.5 * (P + P.T)`) after any update. The Joseph form includes this implicitly. For a 127-frame run this likely has negligible effect, but for longer runs or when many updates happen per second in live mode, it can cause eigenvalue drift. Adding `self.P = 0.5 * (self.P + self.P.T)` after each update is a one-line fix that costs essentially nothing.

---

## CP-34 — Wind is hard-clamped to zero

**Severity:** Medium for real drone, Low for MSFS-only prototype  
**File:** `ekf_ins.py`

### Evidence

Inside `update_airspeed()`:

```python
max_wind = 0.0
```

### Why this matters

This is acceptable for a controlled simulator assumption, but invalid for real outdoor UAV deployment.

### Suggested fix

Move to config:

```python
ASSUME_ZERO_WIND = True
MAX_WIND_MPS = 0.0
```

For real drone tests, allow nonzero wind and estimate it.

### Validation test

Run one simulated wind scenario and verify the EKF does not force wind to zero when `ASSUME_ZERO_WIND=False`.

---

## CP-35 — Particle filter randomness is not reproducible

**Severity:** Medium  
**File:** `particle_filter.py`

### Evidence

`np.random.default_rng()` is created without a seed in initialization, prediction, and resampling.

### Why this matters

Two runs with the same data can differ. That makes debugging and thesis result reproduction harder.

### Suggested fix

Add optional seed/config:

```python
PF_RANDOM_SEED = 42
```

Create `self.rng` once in `ParticleFilter.__init__()` and use it everywhere.

### Validation test

Run file mode twice with the same seed. `results.csv` should be identical.

> **[Claude Code review]:** Confirmed, and worse than described. In `particle_filter.py`, `np.random.default_rng()` is called with no seed in three separate places: `_init_particles()`, `predict()`, and `resample()`. Each call creates an independent Generator from OS entropy. This means not only are two separate runs different, but within a single run, the prediction and resampling use unrelated random streams from the initialization. The fix is to create `self.rng = np.random.default_rng(seed)` once in `__init__` and pass it as a parameter to the three methods, or access it as `self.rng`. The optional `PF_RANDOM_SEED = None` in config (None = non-reproducible, integer = seeded) is the right design.

---

## CP-36 — Particle-filter update uses heading likelihood even though the visual measurement has no independent heading

**Severity:** Medium  
**File:** `particle_filter.py`, `temporal_searcher.py`

### Evidence

PF measurement uses:

```python
"heading": imu_data["heading"]
```

Then `ParticleFilter.update()` applies a heading likelihood against that value.

### Why this matters

The visual measurement gives position, not heading. Feeding IMU heading as the measurement heading can make the update look more informative than it really is.

### Suggested fix

Allow position-only measurements:

```python
{"position": homo_tile_pos, "score": inlier_score, "heading": None}
```

Then skip heading likelihood when heading is `None`.

### Validation test

Compare PF convergence with heading likelihood enabled vs disabled on the same run.

> **[Claude Code review]:** Confirmed in principle, but I rate the practical effect smaller than implied. The heading likelihood uses `MEASUREMENT_NOISE_HEADING_DEG = 15.0` (from config). At 15° sigma, a particle 30° off-heading has likelihood `exp(-30² / (2 × 15²)) ≈ 0.13`, and a particle 60° off has `exp(-60²/450) ≈ 0.00003`. So it does penalize particles significantly for heading mismatch — but the heading is IMU-derived and likely accurate to within a few degrees, so most particles won't be far off in heading. The heading measurement is essentially anchoring the PF heading to the EKF heading, which is circular but not harmful when the EKF heading is accurate (it usually is, as heading is measured directly). The more important concern is that when the EKF heading is wrong (e.g., during a misalignment event), the heading likelihood would reject correct particles that have drifted to the right position but wrong heading. Position-only measurements are cleaner in principle. This is a real design issue but low impact for normal flight conditions.

---

## CP-37 — Search-radius logic is labeled as PF-guided but currently depends on a fixed config tile size

**Severity:** Medium  
**Files:** `temporal_searcher.py`, `config.py`, `particle_filter.py`

### Evidence

`search_radius_m` uses:

```python
region["radius_tiles"] * self.cfg.TILE_SIZE_METERS
```

But the PF itself already has latitude-dependent `_tile_size_m`.

### Why this matters

The config value is computed at latitude `55.6°`, while actual flights may not always be at that latitude. For Denmark this is small, but the pipeline is less general than it looks.

### Suggested fix

Return `radius_m` directly from `ParticleFilter.get_search_region()`.

### Validation test

Compare search radius at different latitudes and ensure it remains physically meaningful.

> **[Claude Code review]:** Confirmed. The fix is one line in `ParticleFilter.get_search_region()`: add `"radius_m": radius_m` to the returned dict, and use that in temporal_searcher instead of the tile multiplication. For Denmark (55.4–57°N), the error between the config tile size and the latitude-dependent tile size is less than 0.5%, so this has no measurable effect in practice. Still a clean consistency fix that costs nothing.

---

## CP-38 — Optional output writing is duplicated and tangled into runtime logic

**Severity:** Medium  
**File:** `run_pipeline.py`

### Evidence

Both file and live modes contain repeated blocks for:

- query JPEG output,
- IMU JSON output,
- PX4 GPS input CSV,
- analysis extras,
- timing rows,
- pipeline trace.

### Why this matters

Output code is not algorithm code. Keeping it inline makes the runtime path harder to verify.

### Suggested fix

Create:

```python
class RunOutputWriter:
    write_result_row(...)
    write_query_frame(...)
    write_imu_row(...)
    write_px4_row(...)
    write_trace(...)
    close()
```

### Validation test

Enable/disable each flag independently and verify expected files appear with correct schemas.

---

## CP-39 — `copy` is imported in `run_pipeline.py` but unused

**Severity:** Low  
**File:** `run_pipeline.py`

### Why this matters

Tiny issue, but it signals that cleanup has not been done.

### Suggested fix

Remove the import.

> **[Claude Code review]:** Confirmed. `import copy` is at line 6 of `run_pipeline.py`. The `copy` module is used in `simconnect_adapter.py` (for `copy.copy(self._latest_row)`) but not in `run_pipeline.py` itself. Remove the import.

---

# 4. Potential problems

These are not proven failures in the uploaded run, but the code structure makes them likely failure points.

---

## PP-01 — Out-of-map operation is not handled as a first-class state

**Severity:** High  
**Files:** `terminal_output.txt`, `meta_tile_builder.py`, `temporal_searcher.py`

### Evidence

The terminal output contains repeated messages like:

```text
MetaTileBuilder: no first-pass tiles for (...) r=...
```

near the end of the live run.

### Why this matters

When the aircraft leaves the reference map, the pipeline falls into repeated visual failure. That is expected, but it should be explicitly detected and logged as `out_of_reference_map`, not treated as generic visual failure.

### Suggested fix

Before tile search, check whether EKF/PF center is within the configured tile bounds plus margin. If outside:

```python
method = "out_of_map_fallback"
visual_available = False
```

### Validation test

Fly across the map boundary and verify mode transition:

```text
temporal_tracking -> map_edge_warning -> out_of_map_fallback
```

---

## PP-02 — High semantic confidence can be misleading in repetitive terrain

**Severity:** Medium  
**Files:** `semantic_confirmer.py`, `semantic_tile_scorer.py`, `results.csv`

### Why this matters

Histogram intersection is intentionally viewpoint-tolerant, but it is also weakly discriminative. Many locations can have similar class distributions, especially in forest/field/suburban regions.

### Suggested fix

Use semantic confidence as one feature, not as trust by itself. Combine it with:

- geometric residual,
- inlier spread,
- temporal consistency,
- tile-rank margin between top-1 and top-2.

### Validation test

Plot semantic confidence against actual homography error. If correlation is weak, do not use semantic confidence aggressively in `R` scaling.

---

## PP-03 — Current map/image domain may still create false-positive matches

**Severity:** High  
**Files:** `results.csv`, `pipeline_trace.pdf`

### Why this matters

MSFS imagery and Bing/reference tiles are visually related but not identical. Local textures such as roads, fields, rooftops, and tree clusters can repeat. SuperPoint+LightGlue can generate many matches in the wrong nearby area.

### Suggested fix

Add a top-K ambiguity check:

```python
top1_score / top2_score
```

or:

```python
top1_matches - top2_matches
```

Reject or de-weight when top-1 is not clearly better.

### Validation test

Save first-pass and second-pass ranked tiles for all high-error accepted frames. Check if wrong frames have low top-1/top-2 margin.

> **[Claude Code review]:** Confirmed. The first-pass tile rankings ARE already saved in trace.json (`first_pass_tiles` list with match counts). The top-1/top-2 margin is computable from those traces without any new instrumentation. For the high-error frames (113–116), checking whether those frames had low margin in the first-pass candidates would immediately validate or invalidate this hypothesis from existing data.

---

## PP-04 — Black padding in rotated query and meta-tiles can influence feature matching and shape confidence

**Severity:** Medium  
**Files:** `visual_measurement.py`, `meta_tile_builder.py`

### Why this matters

Rotating the query expands the canvas and introduces black triangles. Meta-tiles also contain black empty cells when the selected top-K tiles do not form a full rectangle. Features near black boundaries can bias matching and homography shape scoring.

### Suggested fix

Mask invalid/black regions during feature extraction or crop the valid rotated polygon.

### Validation test

Compare match results with and without black-region masking.

---

## PP-05 — The system has no regression baseline stored in code

**Severity:** Medium

### Why this matters

You have good run evidence, but no automated regression test that says: "This version should produce approximately this result on this fixed run."

### Suggested fix

Create `tests/regression/test_live_024_replay.py` using a small subset of saved frames/IMU rows.

Expected checks:

```python
mean_error < threshold
median_error < threshold
gate_pass_rate within range
no crash
```

---

# 5. Assumed problems that need validation

These are not proven from source alone, but they are plausible enough that they deserve targeted tests.

---

## AP-01 — Camera model is probably the main source of systematic offset

**Why assumed:** The code uses both a hard-coded 110 m look-ahead correction and an approximate attitude-shifted center. That strongly suggests the visual measurement is compensating an unmodeled camera projection.

### Suggested validation

Use a set of frames with low homography error and fit residual offset as a function of:

- heading,
- altitude,
- pitch,
- roll,
- speed,
- image crop/resize geometry.

If residuals align with heading, look-ahead is under/overestimated. If residuals rotate with roll/pitch, camera attitude model is the issue.

---

## AP-02 — The EKF may be overconfident after many visual updates

**Why assumed:** Uploaded run shows accepted measurements with large final errors while `ekf_pos_sigma` remains relatively low. The filter may believe it is more certain than it really is.

### Suggested validation

Compute normalized estimation error squared (NEES) when GPS ground truth is available:

```python
nees = error.T @ inv(P_pos) @ error
```

If NEES is consistently above the expected chi-square range, covariance is underestimating real uncertainty.

> **[Claude Code review]:** Confirmed from live_022 data. At frame 126, pos_sigma=42.88m while actual EKF error=337m. NEES = (337)² / (42.88)² ≈ 61.7, versus chi-square(2, 0.99) ≈ 9.2. The filter is roughly 6× overconfident at that point. The NEES analysis is straightforward to implement since GPS ground truth is in the results.csv (`gps_lat`, `gps_lon`). Running it over the full live_022 run would show whether overconfidence is systematic or only occurs during rejection streaks.

---

## AP-03 — Semantic prefilter may occasionally remove the correct tile

**Why assumed:** Semantic prefilter keeps only top-K semantic candidates before SuperPoint. If the segmentation is wrong or reference prediction colors are corrupted, the correct tile can be removed before feature matching.

### Suggested validation

For selected frames, run with:

```python
SEMANTIC_PREFILTER_ENABLED = True
SEMANTIC_PREFILTER_ENABLED = False
```

Compare:

- candidate list,
- correct tile rank,
- runtime,
- final error.

---

## AP-04 — Results depend on random particle initialization

**Why assumed:** PF uses unseeded randomness.

### Suggested validation

Run file replay 10 times with the same input and record mean/median error variance. If output changes materially, seed the PF.

> **[Claude Code review]:** Confirmed and directly testable since file mode provides deterministic input. Three separate `np.random.default_rng()` calls (one each in `_init_particles`, `predict`, `resample`) guarantee different behavior on every run. Seeding is a one-day fix (see CP-35 for details).

---

# 6. Immediate fix priority

Do not try to fix everything at once. That would be messy. Use this order.

| Priority | Fix | Why first |
|---:|---|---|
| 1 | Create canonical pixel-to-lat/lon conversion and remove old inconsistent conversions. | Geometry correctness comes before tuning. |
| 2 | Refactor shared runtime visual-fusion core used by both file and live mode. | Prevents future double-fix bugs. |
| 3 | Add explicit gate columns: geometry, PF innovation, semantic, EKF innovation, EKF update. | Makes debugging possible. |
| 4 | Move look-ahead and camera assumptions into config and log them in `run_meta.json`. | Makes runs reproducible. |
| 5 | Add NIS-based innovation gate or at least stricter logged innovation metrics. | Current gate accepts high-error frames. |
| 6 | Use PF center for temporal search, or prove with an experiment that EKF center is better. | Current code says PF-guided but uses EKF center. |
| 7 | Add paired frame-row packet in SimConnect source. | Reduces time-alignment error. |
| 8 | Add regression replay test from saved frames. | Protects against breaking a working prototype. |
| 9 | Cache tile loads and prediction loads. | Helps latency without changing algorithm. |
| 10 | Seed particle filter for reproducible replay. | Makes thesis results repeatable. |

> **[Claude Code review]:** This priority ordering is reasonable. One adjustment: Priority 3 (gate columns) is partially done by Phase R (`visual_quality_pass`, `ekf_update_applied`). The remaining gap is `pf_innovation_pass` (invisible in current CSV, see CP-09) and the `visual_rejected_reason` overwrite bug. Those specific gaps should be fixed before adding more columns. Also: Priority 5 (NIS gate) should come after validating the simpler fixes to the innovation formula (lower the 150m floor, tighten coefficients) — full NIS requires significant refactoring of the gate to access H and R matrices.

---

# 7. Minimum validation suite before thesis freeze

## 7.1 Geometry tests

- `test_single_tile_pixel_to_latlon_corners()`
- `test_metatile_pixel_to_latlon_corners()`
- `test_visual_measurement_uses_same_converter_as_position_estimator()`

## 7.2 Runtime consistency tests

- file mode and live-replay mode give same results on the same data.
- no duplicate `step_ekf()` on first file-mode row.
- result CSV columns match `RESULT_COLUMNS`.

## 7.3 Gate diagnostics tests

Every frame should log:

```text
geometry_quality_pass
pf_innovation_pass
semantic_gate_pass
ekf_innovation_pass
ekf_update_applied
reject_reason_combined
```

## 7.4 Reproducibility tests

- same file replay + same PF seed = identical result rows.
- `run_meta.json` contains config snapshot and model/map identifiers.

## 7.5 Performance tests

- mean latency,
- P90 latency,
- semantic inference time,
- meta-tile time,
- homography time,
- tile cache hit rate.

---

# 8. Thesis wording recommendation

Use honest wording like this:

> The implemented pipeline should be interpreted as a research prototype for GPS-denied visual localization rather than as a flight-certified navigation system. The system demonstrates that semantic map context, local feature matching, homography-based position extraction, and EKF fusion can produce plausible online localization estimates in a simulated environment. However, the current implementation remains sensitive to image-domain mismatch, camera-viewpoint assumptions, visual false positives, and processing latency. These limitations are explicitly evaluated through gate-health analysis, innovation checks, per-frame trace inspection, and comparison against simulator GPS ground truth used only for benchmarking.

Do **not** write that the system is robust, real-time, or deployment-ready unless the latency and false-positive acceptance issues are fixed.

---

# 9. Bottom line

The current pipeline is a strong thesis prototype, but it is held together by several manually tuned assumptions:

- hard-coded 110 m look-ahead,
- approximate camera/nadir correction,
- loose innovation gate,
- visual quality gate based mainly on CShape and inliers,
- duplicated runtime logic,
- inconsistent legacy coordinate converters,
- non-atomic SimConnect frame/row handling.

The most dangerous technical issue is **wrong-but-plausible visual updates being accepted into the EKF**. The most dangerous software issue is **duplicated live/file runtime fusion logic**.

Fix those two first.

---

# 10. Additional findings from Claude Code source review

These findings were not in the original audit. They come from reading the current source code.

---

## N-01 — `pf_center` in trace.json is mislabeled — it is the EKF position

**Severity:** Medium  
**File:** `temporal_searcher.py:347`

### Evidence

```python
_trace_data['pf_center'] = (center_lat, center_lon)
```

`center_lat, center_lon` is set two lines earlier:

```python
center_lat, center_lon = imu_data["lat"], imu_data["lon"]
```

This is the EKF/IMU position. The actual PF weighted mean is `region["center"]` from `get_search_region()`, which is never stored in trace.

### Why this matters

Every `trace.json` has a `pf_center` field used for analysis and visualizations, but it is actually the EKF position. Anyone reading the trace to understand PF behavior is looking at the wrong value. This directly affects offline analysis of the live_022 and any future run.

### Fix

```python
pf_center_tile = region["center"]
pf_center_lat, pf_center_lon = tile_to_latlon(
    pf_center_tile[0], pf_center_tile[1], self.cfg.TMS_ZOOM_LEVEL)
_trace_data['pf_center'] = (pf_center_lat, pf_center_lon)
_trace_data['ekf_center'] = (center_lat, center_lon)
```

---

## N-02 — Homography winner score uses absolute inlier count, not inlier ratio

**Severity:** Medium  
**File:** `visual_measurement.py:303-307`

### Evidence

```python
def score(branch: Dict) -> float:
    n = branch["inliers"]          # absolute count
    cs = branch["CShape"]
    bonus = 1.5 if branch["convex"] else 1.0
    return n * cs * bonus
```

A noisy match with 600 inliers out of 2000 correspondences (30% ratio, CShape=0.4) scores 360. A clean match with 150 inliers out of 180 correspondences (83% ratio, CShape=0.8) scores 180. The noisy match always wins, even though the high-ratio match is geometrically better.

### Fix

```python
def score(branch: Dict) -> float:
    n = branch["inliers"]
    total = branch.get("n_correspondences") or max(n, 1)
    inlier_ratio = n / total
    cs = branch["CShape"]
    bonus = 1.5 if branch["convex"] else 1.0
    return inlier_ratio * cs * bonus
```

Note: `n_correspondences` is already in the `result` dict at the top of `compute_dual_homography()`, so it is available.

---

## N-03 — PF creates three independent unseeded RNGs per frame

**Severity:** Medium  
**File:** `particle_filter.py:91, 107, 171`

### Evidence

```python
# _init_particles():
rng = np.random.default_rng()   # line 91

# predict():
rng = np.random.default_rng()   # line 107 — new rng every call

# resample():
rng = np.random.default_rng()   # line 171 — new rng every call
```

Each call to `predict()` (once per frame) creates a fresh Generator from OS entropy. Each call to `resample()` (once per frame when N_eff is low) creates another. This is worse than a single unseeded RNG: the three streams are unrelated, and the prediction randomness is reset every frame.

### Fix

```python
def __init__(self, ..., seed=None):
    self.rng = np.random.default_rng(seed)
    ...
```

Use `self.rng` in all three methods. Add `PF_RANDOM_SEED = None` to config (None = non-reproducible for live, set an integer for file-mode replay).

---

## N-04 — `_process_one_frame` double-applies the EKF position update on relocalization frames

**Severity:** Medium  
**File:** `run_pipeline.py:494-537`

### Evidence

During relocalization (`relocalization_applied=True`), the EKF update is applied inside the relocalization block:

```python
ekf.update_position(homo_pos[0], homo_pos[1],
                    R_pos_m2=cfg_r.RELOCALIZATION_R_M ** 2)  # update #1
gate_pass = True
relocalization_applied = True
```

Then later:

```python
if gate_pass and homo_pos is not None:
    if relocalization_applied:
        r_used = searcher.cfg.RELOCALIZATION_R_M ** 2
        ekf_update_applied = True
        # NOTE: no second ekf.update_position() here — correct
    else:
        ...
        ekf.update_position(...)   # only for normal path
```

The second block correctly skips the EKF update for relocalization (it just records `r_used` for logging). So this is NOT a double-update bug as currently written — the update only happens once inside the relocalization block.

### Why noting this

The structure is fragile: the `gate_pass = True` inside the relocalization block means the outer `if gate_pass` runs, and a reader could mistake the intent. A future change that adds `ekf.update_position()` to the outer block without noticing the relocalization branch would cause a double update. The logic should be made explicit:

```python
if gate_pass and homo_pos is not None and not relocalization_applied:
    # Normal path EKF update only
    ...
    ekf.update_position(...)
```

This makes the exclusion of relocalization from the outer update explicit rather than implicit.
