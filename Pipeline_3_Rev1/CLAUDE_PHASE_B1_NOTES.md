# Phase B1 — Implementation Notes

## Environment
- **Venv**: `.final_Pipeline_venv` at `C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\.final_Pipeline_venv`
- **Python**: `C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\.final_Pipeline_venv\Scripts\python.exe`
- **CUDA**: Available — NVIDIA GeForce RTX 5050 Laptop GPU
- **torch.cuda.is_available()**: True

## Checkpoint
- Backup created: `src_backup_phase_a/`, `config_backup_phase_a/`, notebook backup
- Phase A outputs preserved at `outputs/phase_a/` (untouched)
- New outputs go to `outputs/phase_b1/`

## Phase A Baseline (10 test frames)
- Frame indices: [0, 33, 66, 99, 132, 166, 199, 232, 265, 299]
- EKF baseline error: mean=192.9m, median=194.0m
- Unrotated matching: mean 83.7 top-1 matches
- Current homography error (unrotated): mean 354.3m (9/10 valid)

---

## Progress Log

### 2026-04-07 13:40 — Starting Phase B1

**Plan:**
1. Create `src/visual_measurement.py` — new module for:
   - Query heading rotation
   - Dual homography (MAGSAC + DLT as explicit branches)
   - Visual measurement extraction (5 methods)
   - Shape confidence scoring
2. Create `scripts/phase_b1_diagnostics.py` — test on same 10 frames
3. Wire into notebook

**Files to create:**
- `src/visual_measurement.py` (NEW)
- `scripts/phase_b1_diagnostics.py` (NEW)

**Files to modify:**
- None yet — diagnostics first, then integration

### 2026-04-07 — Diagnostics Complete

**Created files:**
- `src/visual_measurement.py` — rotate_image, compute_dual_homography, extract_visual_measurements, compute_shape_confidence
- `scripts/phase_b1_diagnostics.py` — independent per-frame evaluation
- `scripts/analyze_b1.py` — strategy simulation
- `scripts/analyze_b1_strategy.py` — runtime strategy selection
- `scripts/phase_b1_validate.py` — full pipeline end-to-end validation

**Modified files:**
- `src/temporal_searcher.py` — heading rotation before MetaTileBuilder, dual homography, visual measurement extraction, quality-gated blending
- `notebooks/test_temporal_pipeline.ipynb` — added pitch/roll to imu_data

**Key Results (independent per-frame diagnostics):**
- Rotation: 4.3x more matches (83.7 → 363.4 top-1)
- MAGSAC wins 7/10 rotated, DLT wins 3/10
- Best measurement method: `inlier_centroid` (chosen 5/10), `trimmed_centroid` lowest mean
- Quality-gated strategy (CShape>0.3, inliers>20): **138.0m mean** vs 192.9m EKF = **-54.9m (28.5%)**
- Oracle (best of all methods): **149.5m mean**, 8/10 beat EKF

**Pipeline Integration Results (sparse 10-frame validation):**
- Only 1/10 frames activated quality gate (Frame 1: 60.3m from 200.0m)
- Pipeline mean: 178.9m (−14.0m vs EKF)
- Limitation: sparse frames → large dt between frames → particle filter drift → wrong search center

**Known Issue: Performance**
- Rotated images ~1800×2200 vs original 1920×1079
- Frame processing time: ~12s (vs ~3s before rotation)
- 300 frames would take ~60 minutes

---

## Requirements Audit (2026-04-07, live pipeline fix)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | `.final_Pipeline_venv` active | DONE |
| 2 | CUDA actually used | DONE |
| 3 | Notebook is main entrypoint | DONE |
| 4 | Live pipeline runs from notebook | PARTIALLY — frame 0 cold-start has no Phase B1 logic |
| 5 | Code modular and organized | DONE |
| 6 | Heading rotation integrated | PARTIALLY — only `_process_frame_N`; frame 0 has none |
| 7 | DLT branch real | DONE |
| 8 | MAGSAC branch real | DONE |
| 9 | Dual homography winner used | PARTIALLY — only `_process_frame_N`; frame 0 old single-homography |
| 10 | All measurement methods considered | DONE OFFLINE — 5 in module, cascade in N uses 4, frame 0 uses old |
| 11 | Best measurement logic used at runtime | PARTIALLY — `_process_frame_N` has cascade; frame 0 uses BFS old |
| 12 | Good visual frames influence later frames | CRITICAL GAP — gate-fail fallback is raw EKF, ignores PF carrying visual corrections |
| 13 | Notebook reconnected to modules | MOSTLY — stale `estimate_position` import |
| 14 | No giant file | DONE |
| 15 | Local notes updated | DONE |
| 16 | New outputs separate, old preserved | DONE |
| 17 | No GT in runtime decisions | DONE |
| 18 | Live pipeline being fixed | IN PROGRESS |

### Gaps to Fix
- **GAP A**: Cold-start (frame 0) has ZERO Phase B1 improvements (rotation, dual hom, measurements)
- **GAP B**: Quality-gate fallback uses raw EKF instead of PF estimate (kills propagation of visual corrections)
- **GAP C**: No resize after rotation → 12s/frame instead of ~5s
- **GAP D**: nadir_corrected excluded from measurement cascade
- **GAP E**: Stale `estimate_position` import

---

## Fix Session Progress (2026-04-07)

