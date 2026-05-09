# CODEMAP — All_In_One_Pipeline

> Reader's index. Every folder, every entry point, every key file — what they do, who reads them. Source-grounded. Updated 2026-05-09.

For depth on the algorithm itself, see `docs/pipeline_breakdown.tex`.
For thesis-critical GPS-denied integrity, see `docs/GPS_DENIED_INTEGRITY_AUDIT.md` and `docs/BS_CHECK.md`.

---

## 1. Repository top level

```
All_In_One_Pipeline/
├── Pipeline_3_Rev1/                     ← live online localisation pipeline (PRIMARY DELIVERABLE)
├── Dataset_Preprocessing/               ← offline reference-map preparation
├── SemanticTerrainSegmentationModel/    ← semantic-segmentation training + active checkpoint
├── SEMANTIC BEFORE/                     ← frozen Phase 1 winter checkpoint (load-bearing — see § 4)
├── REFERENCE_MAP_ODENSE/                ← active reference map (aerial + prediction + h5)
├── REFERENCE_MAP_VEJLE_20260321_162024/ ← legacy reference map (matches Logs_Run_…)
├── REFERENCE_MAP_CPH/                   ← legacy reference map (h5 only)
├── Logs_Run_20260321_162024/            ← recorded MSFS flight (Vejle/CPH terrain)
├── WMM2025COF/                          ← magnetic-declination coefficients
├── QGIS/                                ← external map-extraction tooling (not part of runtime)
├── TMS_Map_Reconstruction_Check/        ← external map-visualisation utility (not part of runtime)
├── ALL_Docs_from_all/                   ← legacy / archived earlier pipelines
├── live_010.kml, live_011_cph.kml       ← orphan KML exports
├── README.md                            ← top-level entry (re-read after this Phase 1)
├── CLAUDE.md                            ← detailed dev log (referenced from this CODEMAP)
└── .final_Pipeline_venv/, .venv/        ← Python environments
```

---

## 2. Active vs. inactive folders

| Folder | Role | Status |
|---|---|---|
| `Pipeline_3_Rev1/` | Live online localisation pipeline. The only folder that runs at flight time. | **Active — primary deliverable.** |
| `Dataset_Preprocessing/` | Offline reference-map preparation: produces semantic prediction tiles + `reference_features.h5`. | **Active**, run rarely. |
| `SemanticTerrainSegmentationModel/` | Training + active `best.pth`. | **Active**. |
| `SEMANTIC BEFORE/` | Frozen Phase-1 winter checkpoint. The active training notebook resumes from it. **Mis-named but load-bearing.** | **Frozen reference — do not delete.** |
| `REFERENCE_MAP_ODENSE/` | Currently-active reference map (config points here). | **Active**. |
| `REFERENCE_MAP_VEJLE_*/` | Legacy reference map. Matches the recorded `Logs_Run_*` log. | Inactive. |
| `REFERENCE_MAP_CPH/` | Legacy. Only `reference_features.h5` remains. | Inactive. |
| `Logs_Run_20260321_162024/` | Recorded MSFS flight (Vejle/CPH coordinates). | Inactive replay data — does **not** match the active Odense map. |
| `WMM2025COF/` | NOAA magnetic-declination coefficients. | Active. |
| `QGIS/` | External tooling: `QGIS_Centroid_Coordinate_Extractor.py`, layer source extracts, QMetaTiles settings screenshots. **Used by the user before the pipeline runs.** | External utility. |
| `TMS_Map_Reconstruction_Check/` | External: visual area inspection for flight-path planning, after dataset extraction from QGIS. **Not invoked by the pipeline.** | External utility. |
| `ALL_Docs_from_all/old_pipeline/` | Legacy: earlier `dedode_localization_project`, `IMU_Pipeline_Final_old`, `MSFS2020_IMU_Pipeline`. No active imports. | **Legacy archive.** |
| `live_*.kml` (top-level) | Orphan KML exports from earlier runs. | Orphan; safe to ignore. |

---

## 3. Entry points

| # | Command | Mode | Confidence | Output |
|---|---|---|---|---|
| 1 | `python -m runtime.run_pipeline --source simconnect --run-id <id>` | Live runtime | **Confirmed working** ([live_020_Odense_f1](../outputs/runs/live_020_Odense_f1/run_meta.json), 125 frames, 96 % gate pass) | `outputs/runs/<id>/{results.csv,run_meta.json,…}` |
| 2 | `python -m runtime.run_pipeline --source file --run-id <id>` | File replay | **Broken as configured** (recorded log over Vejle, active map is Odense — see [`CURRENT_BEHAVIOUR_BASELINE.md`](CURRENT_BEHAVIOUR_BASELINE.md)) | same |
| 3 | `python -m Dataset_Preprocessing.preprocess_reference --all` | Offline preprocessing | Confirmed working (`REFERENCE_MAP_ODENSE/{prediction/, reference_features.h5}` exist, generated 2026-05-09) | reference-map artefacts |
| 4 | Open `SemanticTerrainSegmentationModel/Semantic_Model_QGIS_8_Class_Rev6.ipynb` and run all | Training | Confirmed working but **notebook-only** (no `.py`) | `best.pth` (latest 2026-03-18) |
| 5 | `python -m Pipeline_3_Rev1.analysis.evaluate_run`, `…export_kml`, `…plot_diagnostics`, `…plot_trajectory` | Post-run evaluation | Confirmed working | analysis figures, KML files |
| 6 | Open `Pipeline_3_Rev1/notebooks/{live_analysis,pipeline_trace,diagnostics}.ipynb` | Debug/eval | Working (`live_analysis`, `pipeline_trace` recently fixed) | inline analysis figures |
| 7 | `pytest Pipeline_3_Rev1/tests -q` | Unit tests | Confirmed working (6 tests covering EKF, PF, meta-tile builder, semantic confirmer, temporal searcher, units) | test pass/fail |

---

## 4. `SEMANTIC BEFORE/` — clarification

The folder name suggests legacy. It is not. `SemanticTerrainSegmentationModel/Semantic_Model_QGIS_8_Class_Rev6.ipynb` (the active training notebook) Phase-2 fine-tuning resumes from:

```
C:\Users\emilj\Documents\Thesis\TRAINING\runs\20260304_222309\best.pth
```

A copy of that exact checkpoint is mirrored in:

```
SEMANTIC BEFORE/1BEST_TRAINING_OUTCOME_20260304_222309/{20260304_222309/, BACKUP MODEL/}
```

If the external `TRAINING/runs/` directory is lost, training-from-scratch (winter → summer) cannot be reproduced. The mirror in `SEMANTIC BEFORE/` is the disaster-recovery copy.

**Action:** do not rename or delete this folder until the training notebook's hard-coded path is rewritten to read from a portable location. Listed for future cleanup in the architecture refactor plan; **not in Phase 0/1 scope**.

---

## 5. `Pipeline_3_Rev1/` map

```
Pipeline_3_Rev1/
├── runtime/
│   ├── run_pipeline.py        (959 L)  ← top-level CLI; both modes; contains GP8 fallback
│   └── simconnect_adapter.py  (338 L)  ← FileSource + SimConnectLiveSource
│
├── src/
│   ├── ekf_ins.py             (775 L)  ← Error-State EKF (10D) + step_ekf + batch wrapper
│   ├── temporal_searcher.py   (631 L)  ← per-frame orchestrator (cold-start vs temporal)
│   ├── visual_measurement.py  (548 L)  ← rotation + dual homography + 5 measurement methods
│   ├── meta_tile_builder.py   (351 L)  ← two-pass tile search + meta-tile + verify
│   ├── particle_filter.py     (281 L)  ← bootstrap PF + resample
│   ├── tile_utils.py          (209 L)  ← TMS math + TileLoader
│   ├── wmm_declination.py     (180 L)  ← WMM2025 lookup
│   ├── best_first_search.py   (158 L)  ← cold-start search
│   ├── semantic_tile_scorer.py (155 L) ← histogram pre-filter
│   ├── position_estimator.py  (147 L)  ← homography → GPS
│   ├── geometric_matcher.py   (119 L)  ← SuperPoint + LightGlue wrapper
│   ├── semantic_model.py      (111 L)  ← UNet++ inference wrapper
│   ├── semantic_confirmer.py  ( 93 L)  ← centroid alignment
│   ├── image_utils.py         ( 65 L)  ← load_image + preprocess_query_frame
│   └── trajectory_smoother.py          ← post-processing (NOT used at runtime)
│
├── config/
│   └── config.py              (260 L)  ← ALL paths, magic numbers, flags
│
├── analysis/
│   ├── evaluate_run.py                 ← post-run accuracy metrics (haversine vs GT)
│   ├── export_kml.py                   ← KML trajectory export
│   ├── plot_diagnostics.py             ← per-frame diagnostic plots
│   └── plot_trajectory.py              ← 2D trajectory plot
│
├── notebooks/
│   ├── live_analysis.ipynb     (558 KB) ← 15-cell analysis of any run (recently fixed)
│   ├── pipeline_trace.ipynb    (8.3 MB) ← per-frame trace viewer (recently fixed)
│   ├── diagnostics.ipynb       (40 KB)  ← smaller diagnostic
│   └── test_temporal_pipeline.ipynb     ← QUESTIONABLE — dev scratchpad with deleted cells
│
├── tests/
│   ├── test_10d_ekf.py
│   ├── test_meta_tile_builder.py
│   ├── test_particle_filter.py
│   ├── test_semantic_confirmer.py
│   ├── test_temporal_searcher.py
│   └── test_units.py
│
├── docs/
│   ├── pipeline_breakdown.tex          ← full LaTeX architecture document
│   ├── CODEMAP.md                      ← THIS FILE
│   ├── GPS_DENIED_INTEGRITY_AUDIT.md   ← top-priority audit
│   ├── BS_CHECK.md                     ← brutally honest assessment
│   ├── CURRENT_BEHAVIOUR_BASELINE.md   ← Phase 0 baseline
│   ├── CALL_GRAPH.md                   ← live runtime call graph
│   ├── ARTEFACT_FLOW.md                ← producer/consumer table
│   ├── Diagrams/00_..06_*.mmd          ← Mermaid diagrams (Phase 1)
│   ├── PIPELINE_07_04_2026.md          ← older Markdown architecture doc
│   ├── PHASE_B1_REPORT.md              ← phase-specific dev notes
│   ├── CLAUDE_PHASE_B1_NOTES.md
│   ├── FLAGS.md                        ← config-flag reference
│   └── altitude_bug.md                 ← bug investigation
│
└── outputs/
    └── runs/<run_id>/
        ├── results.csv                 ← always written (31 cols)
        ├── run_meta.json
        ├── flight_data/                ← when SAVE_QUERY_FRAMES or SAVE_IMU_ROWS
        ├── px4_gps_input.csv           ← when SAVE_ANALYSIS_DATA
        ├── analysis_extras.csv         ← when SAVE_ANALYSIS_DATA
        ├── timing_data.csv             ← when SAVE_TIMING_DATA
        └── pipeline_data/frame_NNNN/   ← when SAVE_PIPELINE_TRACE
```

### Live runtime path (one frame)

```
runtime/run_pipeline.py::run_simconnect_mode
  → step_ekf(row)                       (src/ekf_ins.py)
  → SimConnectLiveSource.get_latest_frame
  → TemporalSearcher.process_frame      (src/temporal_searcher.py)
       → MetaTileBuilder.run               (src/meta_tile_builder.py)
            → matcher.match (SP+LG)         (src/geometric_matcher.py)
            → FeatureStoreLoader.get_features  (Dataset_Preprocessing/feature_store.py)
       → compute_dual_homography           (src/visual_measurement.py)
       → extract_visual_measurements       (src/visual_measurement.py)
       → semantic_model.predict            (src/semantic_model.py)
       → semantic_confirmer.confirm        (src/semantic_confirmer.py)
       → particle_filter.update / resample (src/particle_filter.py)
  → look-ahead correction (110 m × cos bank)
  → adaptive R = R_base · m_bank · m_verify · m_sem
  → ekf.update_position(homo_corr, R)   (src/ekf_ins.py)        — gate-pass branch
  OR
  → ekf.update_position(sim, R = 200²)                          — GP8 fallback (gate-fail)
  → writer.writerow(result_row)
```

Full call graph: see `docs/CALL_GRAPH.md`.

---

## 6. `Dataset_Preprocessing/` map

```
Dataset_Preprocessing/
├── preprocess_reference.py     (191 L)  ← CLI: --all / --semantic / --superpoint
├── semantic_preprocessor.py    (243 L)  ← aerial → prediction tiles
├── superpoint_preprocessor.py  (203 L)  ← aerial → reference_features.h5
├── feature_store.py            (385 L)  ← HDF5 reader/writer + validate (cleanest file in repo)
├── config.py                   ( 76 L)  ← preprocessing config (mirror of Pipeline_3_Rev1's)
└── __init__.py                 (  1 L)
```

---

## 7. `SemanticTerrainSegmentationModel/` and `SEMANTIC BEFORE/`

```
SemanticTerrainSegmentationModel/
├── Semantic_Model_QGIS_8_Class_Rev6.ipynb  ← active training (notebook-only)
├── best.pth                                 ← active checkpoint (loaded at runtime + offline)
├── latest.pth                               ← resume snapshot
├── epoch_0010.pth, epoch_0020.pth           ← intermediate snapshots
├── train_log.csv                            ← per-epoch metrics
├── per_class_iou.jsonl                      ← combined-split IoU
├── per_class_iou_winter.jsonl               ← winter-split IoU
├── per_class_iou_summer.jsonl               ← summer-split IoU (drives best.pth selection)
├── legend.txt                               ← class definitions (6 classes)
└── config.json                              ← run config (batch=8, patience=15)

SEMANTIC BEFORE/1BEST_TRAINING_OUTCOME_20260304_222309/
├── 20260304_222309/                         ← Phase 1 winter run
│   ├── best.pth                              ← loaded by Phase 2 notebook (resume point)
│   ├── latest.pth, epoch_*.pth
│   ├── train_log.csv
│   └── config.json                           ← patience=30 (the older config)
├── BACKUP MODEL/                             ← exact duplicate of 20260304_222309/
└── Model_Evaluation_Visualization.pdf        ← Phase 1 evaluation report
```

---

## 8. Class definitions and the duplication risk (D8)

The 6 semantic classes are defined in **three places** that must agree:

| Definition site | Used by |
|---|---|
| `SemanticTerrainSegmentationModel/legend.txt` | training notebook |
| Training notebook (`CLASS_NAMES` literal) | training |
| `Pipeline_3_Rev1/config/config.py:82-97` (`SEMANTIC_CLASSES`, `COLOR_MAP`) | runtime + offline preprocessing |

They currently agree (verified 2026-05-09):

```
0: waterbodies   (4, 4, 255)
1: forest_trees  (0, 167, 2)
2: land          (243, 255, 150)
3: railway       (193, 105, 53)
4: roads         (255, 0, 231)
5: buildings     (150, 150, 150)
```

Nothing in the build enforces this. A future cleanup should add a unit test that asserts they agree (cheap, low-risk, deferred to Phase 2+).

---

## 9. RGB → Grayscale duplication risk (D9)

The byte-identical RGB-to-grayscale conversion appears in **two places**:

- `Dataset_Preprocessing/superpoint_preprocessor.py:53-67` (offline tile features)
- `Pipeline_3_Rev1/src/geometric_matcher.py` (runtime query feature extraction)

Both use the same Y-channel weights `[0.2989, 0.5870, 0.1140]`. They must remain bit-identical or runtime-extracted query features will not match precomputed reference features in `reference_features.h5`. Currently held together by hope — no test enforces it. Deferred to Phase 5.

---

## 10. The `sys.path.insert` cross-folder import (D6)

`Pipeline_3_Rev1/runtime/run_pipeline.py:36-38` performs:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent           # Pipeline_3_Rev1/
REPO = ROOT.parent                 # All_In_One_Pipeline/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))
```

This makes `Dataset_Preprocessing.feature_store` importable from `Pipeline_3_Rev1/`. The dependency is invisible to standard import-graph tools and IDEs. Documented for awareness; deferred to Phase 5.

---

## 11. Stale config entry (D7 / P4)

`config/config.py:25` still references:
```python
REFERENCE_METADATA_CSV = ALL_IN_ONE_ROOT / "reference_tiles_metadata.csv"
```
Current preprocessing does **not** generate this file (`TileLoader.list_tiles()` walks directories instead). Documented for awareness; safe to remove in Phase 4.

---

## 12. Where to read next

| You want to … | Read |
|---|---|
| Verify the GPS-denied claim survives audit | [`GPS_DENIED_INTEGRITY_AUDIT.md`](GPS_DENIED_INTEGRITY_AUDIT.md) |
| Decide what to disclose in the thesis | [`BS_CHECK.md`](BS_CHECK.md) |
| Trace the live-mode runtime per frame | [`CALL_GRAPH.md`](CALL_GRAPH.md) |
| See producer→consumer for every artefact | [`ARTEFACT_FLOW.md`](ARTEFACT_FLOW.md) |
| Get the full algorithmic / mathematical write-up | [`pipeline_breakdown.tex`](pipeline_breakdown.tex) |
| See the Mermaid diagrams | [`Diagrams/`](Diagrams/) |
| Run the pipeline | the [top-level README](../../README.md) |
| Understand the bug history | [`../../CLAUDE.md`](../../CLAUDE.md) |
| Decide which output flags to enable | [`FLAGS.md`](FLAGS.md) |
