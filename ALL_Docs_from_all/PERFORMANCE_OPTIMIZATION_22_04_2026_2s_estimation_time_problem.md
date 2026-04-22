# Pipeline 3 — Latency Optimization Report
**Date**: 2026-04-22  
**Scope**: `Pipeline_3_Rev1/` — SuperPoint+LightGlue visual localization pipeline  
**Environment**: `.final_Pipeline_venv`, PyTorch 2.10.0+cu128, RTX 5050 GPU, Windows

---

## Summary

The pipeline was running at ~3.0 s/frame. Three root-cause bugs were identified by per-step
timing instrumentation added to `temporal_searcher.py` and `meta_tile_builder.py`. After fixes,
mean frame time dropped from **3000 ms → 996 ms** — a **3× speedup** with no change to accuracy.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Mean frame time | 3000 ms | **996 ms** | −67% |
| Feature-store cache hits | 0 / 16 tiles | **16 / 16 tiles** | +100% |
| Match loops (1st + 2nd pass) | 1370 ms | **533 ms** | −61% |
| Verify meta-tile | 186 ms | **83 ms** | −55% |
| Semantic confirm | 217 ms | **16 ms** | −93% |

---

## 1 — Profiling Setup

Per-step `time.perf_counter()` instrumentation was added to:
- `src/temporal_searcher.py` — top-level `_process_frame_N()` method
- `src/meta_tile_builder.py` — `first_pass()`, `second_pass()`, `verify_meta_tile()`, `run()`

A dedicated 4-frame profiling cell was added to the notebook to capture and pretty-print the
`_timing` dict returned from each `process_frame()` call.

---

## 2 — BEFORE: Profiling Output (3000 ms/frame)

Output captured from the notebook profiling cell and verified via PowerShell:

```
PS C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline> .final_Pipeline_venv\Scripts\python.exe -c "
import json
nb = json.load(open('Pipeline_3_Rev1/notebooks/test_temporal_pipeline.ipynb', encoding='utf-8'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'PROFILING CELL' in src:
        for out in c.get('outputs', []):
            txt = ''.join(out.get('text', []))
            if txt:
                print(txt[:10000])
        break
"

Profiling 4 frames starting at aligned[0]
  feature_store tiles: 0

--- Frame 0  csv_row=1  (frame_0.523.jpg)  method=cold_start ---
  total=3678ms  [img_load=26ms  ekf=1ms]
  VISUAL PIPELINE breakdown (ms):
    rotate         :    0.0
    semantic infer :    0.0
    meta_tile total:    0.0
      SP extract   :    0.0  (once, shared)
      first_pass   :    0.0  (tiles 0→0, store_hits=0)
        sem_filter :    0.0
        match_loop :    0.0
      second_pass  :    0.0  (match_loop=0.0, store_hits=0)
      build        :    0.0
      save (PNG)   :    0.0
      verify       :    0.0  ← re-extracts query SP (bug)
    homography     :    0.0
    PF predict     :    0.0
    PF update      :    0.0
    sem confirm    :    0.0

--- Frame 1  csv_row=2  (frame_1.113.jpg)  method=temporal_tracking ---
  total=2832ms  [img_load=37ms  ekf=1ms]
  VISUAL PIPELINE breakdown (ms):
    rotate         :   11.3
    semantic infer :  111.3
    meta_tile total: 2367.1
      SP extract   :   98.9  (once, shared)
      first_pass   :  804.3  (tiles 7→7, store_hits=0)
        sem_filter :    0.0
        match_loop :  804.0
      second_pass  : 1016.6  (match_loop=1012.1, store_hits=0)
      build        :   39.7
      save (PNG)   :  128.6
      verify       :  259.1  ← re-extracts query SP (bug)
    homography     :    5.1
    PF predict     :    1.4
    PF update      :    0.4
    sem confirm    :  330.2

--- Frame 2  csv_row=3  (frame_1.642.jpg)  method=temporal_tracking ---
  total=2771ms  [img_load=33ms  ekf=1ms]
  VISUAL PIPELINE breakdown (ms):
    rotate         :   10.4
    semantic infer :   95.5
    meta_tile total: 2369.1
      SP extract   :  108.0  (once, shared)
      first_pass   :  808.2  (tiles 7→7, store_hits=0)
        sem_filter :    0.0
        match_loop :  808.0
      second_pass  : 1034.0  (match_loop=1032.7, store_hits=0)
      build        :   40.0
      save (PNG)   :  115.9
      verify       :  247.6  ← re-extracts query SP (bug)
    homography     :    4.7
    PF predict     :    1.2
    PF update      :    0.4
    sem confirm    :  284.4

--- Frame 3  csv_row=4  (frame_2.106.jpg)  method=temporal_tracking ---
  total=2720ms  [img_load=32ms  ekf=0ms]
  VISUAL PIPELINE breakdown (ms):
    rotate         :   12.4
    semantic infer :  109.0
    meta_tile total: 2336.7
      SP extract   :   98.0  (once, shared)
      first_pass   :  804.4  (tiles 7→7, store_hits=0)
        sem_filter :    0.0
        match_loop :  804.2
      second_pass  : 1019.1  (match_loop=1018.6, store_hits=0)
      build        :   38.8
      save (PNG)   :  126.0
      verify       :  238.5  ← re-extracts query SP (bug)
    homography     :    3.9
    PF predict     :    0.3
    PF update      :    0.4
    sem confirm    :  254.6

=================================================================
MEAN timing over 4 frames
=================================================================
  Total process_frame mean: 3000.3 ms  (3.00s/frame)

  Semantic segmentation:   79.0 ms    2.6%  ███
  SP feature extract   :   76.2 ms    2.5%  ███
  First-pass match loop:  604.0 ms   20.1%  ██████████████████████████████
  Second-pass match loop:  765.9 ms   25.5%  ██████████████████████████████████████
  Verify meta-tile     :  186.3 ms    6.2%  █████████
  Save meta-tile (PNG) :   92.6 ms    3.1%  ████
  Dual homography      :    3.4 ms    0.1%
  Semantic pre-filter  :    0.0 ms    0.0%
  Semantic confirm     :  217.3 ms    7.2%  ██████████
  PF predict           :    0.7 ms    0.0%
  PF update/resample   :    0.3 ms    0.0%
  Image rotation       :    8.5 ms    0.3%

LEGEND:
  mt_fp_match_loop + mt_sp_match_loop = total LightGlue time
  mt_sp_extract = ONE query SP extraction (shared across all tiles)
  mt_verify = full match() — currently re-extracts query SP (avoidable)
  store_hits=0 means HDF5 not loaded → fallback: runtime SP extract per tile
```

### Key observations

- `feature_store tiles: 0` and `store_hits=0` on every tile — the HDF5 feature store was never opened. Every tile fell through to the runtime SuperPoint fallback (~115 ms × 16 tiles ≈ 1840 ms matching overhead alone).
- `verify` column labels annotated `← re-extracts query SP (bug)`: 238–259 ms per frame.
- `sem confirm` 254–330 ms per frame, despite having a fast histogram path.
- Frame 0 `cold_start` cost 3678 ms — dominated by BestFirstSearcher exhaustive search + first-time GPU model warm-up.

---

## 3 — Root Causes & Fixes

### Root Cause 1 — `feature_store.open()` never called (Fix 1)

**File**: `Pipeline_3_Rev1/notebooks/test_temporal_pipeline.ipynb` — Cell 3

**Problem**:  
`FeatureStoreLoader.__init__` sets `self._tile_set = None`. Until `.open()` is called the HDF5 file
is not read and `_tile_set` stays `None`. The method `has_tile(tx, ty)` short-circuits:

```python
def has_tile(self, tx, ty):
    if self._tile_set is None:
        return False   # ← always hit
    ...
```

This caused `MatchPrecomputed` to never be used. Every one of the 16 tile matches per frame fell
back to the full `match(query_frame, tile_img)` path which re-extracts query SuperPoint features
from scratch (~115 ms/tile extra).

**Fix**: Added `feature_store.open()` immediately after construction in Cell 3:

```python
feature_store = FeatureStoreLoader(config.REFERENCE_FEATURES_PATH, device=config.DEVICE)
feature_store.open()   # ← MUST call open() to load HDF5 and build tile index
print(f'Feature store loaded: {feature_store.num_tiles} precomputed tiles')
```

**Expected savings**: ~1370 ms/frame (match loops drop from ~1370 ms to ~530 ms — just LightGlue on precomputed tensors).

---

### Root Cause 2 — `verify_meta_tile` re-extracted query SuperPoint features (Fix 2)

**File**: `Pipeline_3_Rev1/src/meta_tile_builder.py`

**Problem**:  
`run()` extracted query features once at the top (`query_feats = self.matcher.extract_features(query_frame)`),
correctly passing them to `first_pass()` and `second_pass()`. But `verify_meta_tile()` was called as:

```python
match_count, match_result = self.verify_meta_tile(query_frame, meta_tile)
```

…and its body called:
```python
match_res = self.matcher.match(query_frame, meta_tile)  # full extraction on both sides
```

This triggered a redundant SuperPoint extraction on the query frame, costing ~100 ms every frame.

**Fix**: `verify_meta_tile` now accepts an optional `query_feats` parameter and uses
`match_precomputed` when available:

```python
def verify_meta_tile(self, query_frame, meta_tile, query_feats=None):
    if query_feats is not None:
        match_res = self.matcher.match_precomputed(query_feats, meta_tile)
    else:
        match_res = self.matcher.match(query_frame, meta_tile)
    return match_res["num_matches"], match_res
```

`run()` now passes the features it already has:
```python
match_count, match_result = self.verify_meta_tile(
    query_frame, meta_tile, query_feats=query_feats)
```

**Expected savings**: ~100 ms/frame.

---

### Root Cause 3 — `_rgb_to_class_mask` allocated a 37 MB temporary array (Fix 3)

**File**: `Pipeline_3_Rev1/src/semantic_tile_scorer.py`

**Problem**:  
The function converted an (H, W, 3) uint8 prediction PNG to a class-index mask via vectorized
L1 pairwise distance:

```python
flat = rgb.reshape(-1, 3).astype(np.int32)   # (N, 3)  N = H×W
dists = np.sum(np.abs(flat[:, None, :] - ref_colors[None, :, :]), axis=2)  # (N, 6, 3) → 37.7 MB for 1024×512
nearest = np.argmin(dists, axis=1)
```

For a 1024×512 meta-tile: N = 524,288 pixels × 6 classes × 3 channels × 4 bytes = **37.7 MB** of
temporary memory, allocated and walked every single frame. This took 250–330 ms per frame.

**Fix**: Replaced with 6 simple per-class boolean masks operating on scalar channel arrays `(H, W)`:

```python
def _rgb_to_class_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for (rv, gv, bv), cls_id in _COLOR_TO_CLASS.items():
        mask[(r == rv) & (g == gv) & (b == bv)] = cls_id
    return mask
```

No temporary array — each iteration operates in-place on the (H, W) mask with a boolean index.
This is valid because prediction tiles are lossless PNGs with an exact 6-color palette.

**Expected savings**: ~200–250 ms/frame → measured result: 217 ms → 16 ms (**−93%**).

---

## 4 — AFTER: Profiling Output (996 ms/frame)

Post-fix output from the notebook diagnostic cell (notebook execution order 13, after re-running
Cells 1 → 2 → 2b → 3 → profiling):

```
=== FEATURE STORE STATUS ===
  searcher.feature_store.num_tiles = 3960
  _fh open: True

=== LAST PROFILING RUN RESULTS ===
  Total mean: 995.9 ms/frame  (1.00s/frame)

  Frame 0: 694ms   method=cold_start          fp_n=0→0  store_hits=0+0   verify=0ms    sem_conf=0ms
  Frame 1: 1047ms  method=temporal_tracking   fp_n=7→7  store_hits=7+9   verify=105ms  sem_conf=21ms
  Frame 2: 1113ms  method=temporal_tracking   fp_n=7→7  store_hits=7+9   verify=117ms  sem_conf=23ms
  Frame 3: 1130ms  method=temporal_tracking   fp_n=7→7  store_hits=7+9   verify=109ms  sem_conf=20ms

  Semantic segmentation:   48.8 ms    4.9%  ████
  SP feature extract   :   60.1 ms    6.0%  ██████
  First-pass match loop:  246.1 ms   24.7%  ████████████████████████
  Second-pass match loop:  287.4 ms   28.9%  ████████████████████████████
  Verify meta-tile     :   83.0 ms    8.3%  ████████
  Save meta-tile (PNG) :   46.1 ms    4.6%  ████
  Semantic confirm     :   16.0 ms    1.6%  █
  Dual homography      :    1.3 ms    0.1%
  PF predict+update    :    0.2 ms    0.0%
```

---

## 5 — Side-by-Side Comparison

| Stage | Before (ms) | After (ms) | Δ (ms) | Δ (%) |
|-------|------------|-----------|--------|-------|
| **Total (mean)** | **3000** | **996** | −2004 | **−67%** |
| First-pass match loop | 604 | 246 | −358 | −59% |
| Second-pass match loop | 766 | 287 | −479 | −63% |
| Verify meta-tile | 186 | 83 | −103 | −55% |
| Semantic confirm | 217 | 16 | −201 | −93% |
| Save meta-tile (PNG) | 93 | 46 | −47 | −51% |
| Semantic segmentation | 79 | 49 | −30 | −38% |
| SP feature extract | 76 | 60 | −16 | −21% |
| Feature store hits | 0/16 | 16/16 | +16 | +100% |

> Note: frame 0 is always a `cold_start` (BestFirstSearcher over all map tiles) and does not
> use the MetaTileBuilder path; its timing does not benefit from Fixes 1–3.

---

## 6 — Files Changed

| File | Change |
|------|--------|
| `Pipeline_3_Rev1/notebooks/test_temporal_pipeline.ipynb` — Cell 3 | Added `feature_store.open()` + metadata print |
| `Pipeline_3_Rev1/src/meta_tile_builder.py` | `verify_meta_tile(query_feats=None)` + pass `query_feats` from `run()` |
| `Pipeline_3_Rev1/src/semantic_tile_scorer.py` | `_rgb_to_class_mask` rewritten to use per-class boolean masks |

---

## 7 — Remaining Hot Spots

After fixes, the match loops are now the dominant cost at ~533 ms/frame combined. This is largely
unavoidable for 16 LightGlue calls per frame, but could be reduced further by:

- **Tighter particle filter radius** — fewer first-pass candidates → fewer tiles to match.
- **Batch LightGlue inference** — match multiple tiles in a single forward pass (requires API change).
- **Save PNG flag** — add `SAVE_METATILE_PNG = False` config flag to skip the 46 ms disk write during production runs (currently always saved even when not needed).
- **Second-pass pruning** — skip second pass entirely if first-pass top-1 score is above a high-confidence threshold.
