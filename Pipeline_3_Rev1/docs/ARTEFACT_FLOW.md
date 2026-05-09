# Artefact Flow
**Pipeline 3 — All_In_One_Pipeline**
*Producer → consumer table for every artefact on disk. Source-grounded. Updated 2026-05-09.*

For the diagram form: [`Diagrams/06_artifact_flow.mmd`](Diagrams/06_artifact_flow.mmd).

---

## 1. Visualisation

```
                                 ┌─────────────────────────┐
[NOAA WMM2025]   ──────────────▶│  WMM2025COF/WMM2025.COF │ ──▶ src/wmm_declination.py
                                 └─────────────────────────┘

[QGIS QMetaTiles plugin]   (external; configured per QGIS/QGIS_Qmetatiles_settings.png)
                │
                ▼
┌──────────────────────────────────────────────┐
│ REFERENCE_MAP_ODENSE/aerial/16/{x}/{y}.png    │ ── TileLoader.load_aerial
│        (active reference imagery, 2026-05-09) │ ── Dataset_Preprocessing/{semantic_,superpoint_}preprocessor.py
└──────────────────────────────────────────────┘

[Hand-labelled training data]   (external; C:\…\TRAINING\6Class_Dataset_Zoom_16_Rev3\)
                │
                ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  SEMANTIC BEFORE/1BEST_TRAINING_OUTCOME_20260304_222309/.../best.pth (frozen)   │
│       ↑                                                                         │
│       │ resume target                                                           │
│       │                                                                         │
│  SemanticTerrainSegmentationModel/Semantic_Model_QGIS_8_Class_Rev6.ipynb        │
│         │                                                                       │
│         ▼                                                                       │
│  SemanticTerrainSegmentationModel/best.pth     (active, 2026-03-18)             │
│  + latest.pth, epoch_*.pth, train_log.csv, per_class_iou*.jsonl, config.json    │
└────────────────────────────────────────────────────────────────────────────────┘
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ Dataset_Preprocessing/   │    │ Pipeline_3_Rev1/         │
│   semantic_preprocessor  │    │   src/semantic_model.py  │
│                          │    │   load_semantic_model    │
│ writes prediction tiles  │    │   (runtime inference)    │
└──────────────┬───────────┘    └──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ REFERENCE_MAP_ODENSE/prediction/16/{x}/{y}.png│ ── TileLoader.load_prediction
│       (precomputed terrain class masks)       │ ── SemanticTileScorer (histogram pre-filter)
└──────────────────────────────────────────────┘ ── SemanticConfirmer (centroid alignment)

[Aerial tiles (above)]
        │
        ▼
┌──────────────────────────────────────────────┐
│ Dataset_Preprocessing/superpoint_preprocessor│
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ REFERENCE_MAP_ODENSE/reference_features.h5    │ ── FeatureStoreLoader
│        (precomputed SuperPoint features)      │ ── consumed by MetaTileBuilder fast-path
└──────────────────────────────────────────────┘

──────────────────────────────────────────────────────────────────────────────────────────

[MSFS 2020 SimConnect]  ── runtime/simconnect_adapter.py:SimConnectLiveSource
                        ── runtime/run_pipeline.py::run_simconnect_mode
                                  │
                                  ▼
                        outputs/runs/<run_id>/results.csv
                                  + run_meta.json
                                  + (optional) flight_data/, px4_gps_input.csv,
                                                analysis_extras.csv, timing_data.csv,
                                                pipeline_data/frame_NNNN/

[Logs_Run_*/imu_gps_log_*.csv + images_*/frame_*.jpg]   (recorded MSFS replay)
                        ── runtime/simconnect_adapter.py:FileSource
                        ── runtime/run_pipeline.py::run_file_mode
                                  │
                                  ▼
                        same outputs/runs/<run_id>/* layout
                        (currently broken-as-configured — log over Vejle, map is Odense)
```

---

## 2. Producer / consumer table

Class column: **A** = active, **F** = frozen, **L** = legacy / not used at runtime, **O** = optional (controlled by config flags), **E** = external utility.

| Artefact | Path / pattern | Producer | Consumer | Class |
|---|---|---|---|---|
| Aerial tiles | `REFERENCE_MAP_ODENSE/aerial/16/{x}/{y}.png` | QGIS QMetaTiles plugin (external) | `TileLoader.load_aerial`, `semantic_preprocessor`, `superpoint_preprocessor` | A |
| Aerial tiles (legacy maps) | `REFERENCE_MAP_VEJLE_*/aerial/…`, `REFERENCE_MAP_CPH/…` | same | only legacy `Logs_Run_*` replay | L |
| Hand-labelled training data | `C:\…\TRAINING\6Class_Dataset_Zoom_16_Rev3\` | external labelling | `Semantic_Model_QGIS_8_Class_Rev6.ipynb` | A (external) |
| Phase-1 winter checkpoint | `SEMANTIC BEFORE/1BEST_TRAINING_OUTCOME_20260304_222309/.../best.pth` | training notebook (Phase 1, 2026-03-06) | training notebook (Phase 2 resume) | F (load-bearing) |
| Active checkpoint | `SemanticTerrainSegmentationModel/best.pth` | training notebook (2026-03-18) | `src/semantic_model.py::load_semantic_model` (runtime), `semantic_preprocessor.py` (offline) | A |
| Resume snapshot | `SemanticTerrainSegmentationModel/{latest,epoch_0010,epoch_0020}.pth` | training notebook | training notebook (resume after crash) | A |
| Training log | `SemanticTerrainSegmentationModel/train_log.csv` | training notebook | manual inspection, `per_class_iou` plotting | A |
| Per-class IoU | `SemanticTerrainSegmentationModel/per_class_iou{,_winter,_summer}.jsonl` | training notebook | manual inspection / training-curve plots | A |
| Class definitions | `SemanticTerrainSegmentationModel/legend.txt` | hand-authored | training notebook | A (duplicated — D8) |
| Class definitions | `Pipeline_3_Rev1/config/config.py:82-97` | hand-authored | runtime + offline preprocessing | A (duplicated — D8) |
| Prediction tiles | `REFERENCE_MAP_ODENSE/prediction/16/{x}/{y}.png` | `Dataset_Preprocessing/semantic_preprocessor.py` | `TileLoader.load_prediction`, `SemanticTileScorer`, `SemanticConfirmer` | A |
| Precomputed SP features | `REFERENCE_MAP_ODENSE/reference_features.h5` (3.5 GB) | `Dataset_Preprocessing/superpoint_preprocessor.py` | `Dataset_Preprocessing/feature_store.py::FeatureStoreLoader` (used by `MetaTileBuilder.first_pass`) | A |
| Stale config reference | `<root>/reference_tiles_metadata.csv` | (legacy; not produced) | (referenced in `config.py:25`, never read at runtime) | L (stale config) |
| Recorded query frames | `Logs_Run_*/images_*/frame_*.jpg` | external MSFS log | `FileSource`, `load_image` | L (mismatched against Odense) |
| Recorded IMU CSV | `Logs_Run_*/imu_gps_log_*.csv` | external MSFS log | `FileSource`, `step_ekf` | L (mismatched against Odense) |
| Recorded EKF CSV | `Logs_Run_*/ekf_ins_*.csv` | external (legacy data_logger) | not currently used by Pipeline 3 | L |
| WMM2025 coefficients | `WMM2025COF/WMM2025.COF` | NOAA (external) | `src/wmm_declination.py::get_mag_field` | A |
| QGIS layer source extracts | `QGIS/raw_output_layer_sources_qgis.txt`, `…docx`, `…png`, `QGIS_Centroid_Coordinate_Extractor.py` | QGIS user (external workflow) | reproducibility documentation | E |
| TMS reconstruction tool | `TMS_Map_Reconstruction_Check/` | external utility | flight-path planning, area inspection | E |
| Live-mode results | `Pipeline_3_Rev1/outputs/runs/<run_id>/results.csv` | `run_pipeline.py::run_simconnect_mode` | analysis notebooks, `analysis/evaluate_run.py` | A |
| Live-mode meta | `Pipeline_3_Rev1/outputs/runs/<run_id>/run_meta.json` | same | analysis notebooks | A |
| Pipeline trace | `Pipeline_3_Rev1/outputs/runs/<run_id>/pipeline_data/frame_NNNN/{query.jpg, query_rotated.jpg, semantic_mask.png, reference_tile.png, matches.png, imu.json, trace.json}` | `run_pipeline.py` (when `SAVE_PIPELINE_TRACE=True`) | `notebooks/pipeline_trace.ipynb` | O |
| PX4 GPS_INPUT | `Pipeline_3_Rev1/outputs/runs/<run_id>/px4_gps_input.csv` | `run_pipeline.py` (when `SAVE_ANALYSIS_DATA=True`) | external PX4 autopilot ingestion (MAVLink MSG 232) | O |
| Analysis extras | `Pipeline_3_Rev1/outputs/runs/<run_id>/analysis_extras.csv` | same | `notebooks/live_analysis.ipynb` Cell 13 | O |
| Timing | `Pipeline_3_Rev1/outputs/runs/<run_id>/timing_data.csv` | `run_pipeline.py` (when `SAVE_TIMING_DATA=True`) | `notebooks/live_analysis.ipynb` Cell 14 | O |
| Query frame archive | `Pipeline_3_Rev1/outputs/runs/<run_id>/flight_data/frame_NNNN.jpg` | `run_pipeline.py` (when `SAVE_QUERY_FRAMES=True`) | manual inspection / re-processing | O |
| IMU row archive | `Pipeline_3_Rev1/outputs/runs/<run_id>/flight_data/frame_NNNN_imu.json` | `run_pipeline.py` (when `SAVE_IMU_ROWS=True`) | manual inspection / re-processing | O |
| Trajectory KMLs | `live_010.kml`, `live_011_cph.kml` (top-level) | earlier `analysis/export_kml.py` runs | external Google Earth viewing | L (orphaned at root) |
| Analysis figures | `Pipeline_3_Rev1/outputs/analysis/<run_id>/*.png, *.gif` | analysis notebooks + `analysis/*.py` | thesis chapters | A |

---

## 3. Cross-folder dependency edges

Edges where one folder consumes an artefact produced by another:

| Producer folder | Consumer folder | Artefact | Mechanism |
|---|---|---|---|
| `SemanticTerrainSegmentationModel/` | `Pipeline_3_Rev1/` | `best.pth` | `config.SEMANTIC_MODEL_PATH` → `load_semantic_model` |
| `SemanticTerrainSegmentationModel/` | `Dataset_Preprocessing/` | `best.pth` | `config.SEMANTIC_MODEL_PATH` → `semantic_preprocessor._load_semantic_model` |
| `SEMANTIC BEFORE/` | `SemanticTerrainSegmentationModel/` | `best.pth` (Phase 1 winter) | training notebook hard-coded path → Phase 2 resume |
| `Dataset_Preprocessing/` | `Pipeline_3_Rev1/` | prediction tiles + `reference_features.h5` | filesystem (under `REFERENCE_MAP_ODENSE/`) |
| `Dataset_Preprocessing/feature_store.py` | `Pipeline_3_Rev1/runtime/run_pipeline.py` | `FeatureStoreLoader` Python class | `sys.path.insert(0, REPO)` import — see CODEMAP § 10 |
| `WMM2025COF/` | `Pipeline_3_Rev1/` | `WMM2025.COF` | `src/wmm_declination.py` reads file path |
| `Logs_Run_*/` | `Pipeline_3_Rev1/` | recorded frames + IMU CSV | `config.QUERY_FRAMES_DIR`, `config.IMU_CSV_PATH` (file mode) |

There are **no Python imports** between active code folders other than the documented `sys.path` injection for `feature_store`. All other coupling is via filesystem.

---

## 4. Notes on the live runtime path

A complete `live_NNN_*` run consumes:
- `SemanticTerrainSegmentationModel/best.pth`
- `REFERENCE_MAP_ODENSE/aerial/16/{x}/{y}.png`
- `REFERENCE_MAP_ODENSE/prediction/16/{x}/{y}.png`
- `REFERENCE_MAP_ODENSE/reference_features.h5`
- `WMM2025COF/WMM2025.COF`
- live MSFS SimConnect telemetry + screen capture

and produces:
- `Pipeline_3_Rev1/outputs/runs/<run_id>/results.csv` (always)
- `Pipeline_3_Rev1/outputs/runs/<run_id>/run_meta.json` (always)
- optional outputs per `config/config.py` flags

A complete file-mode run additionally consumes:
- `Logs_Run_*/imu_gps_log_*.csv`
- `Logs_Run_*/images_*/frame_*.jpg`

and produces the same outputs (modulo `inference_ms` always being `None` in file mode).

---

## 5. Deferred clean-ups

- **Legacy reference maps** (`REFERENCE_MAP_VEJLE_*/`, `REFERENCE_MAP_CPH/`): currently consume disk but unused; safe to archive once a thesis-evaluation file-mode log is recorded over Odense.
- **Stale config entry** `REFERENCE_METADATA_CSV` (`config.py:25`): remove in Phase 4.
- **Top-level orphan KMLs** (`live_010.kml`, `live_011_cph.kml`): move into `outputs/exports/` in Phase 3 (deferred).
- **`SEMANTIC BEFORE/` folder name**: leave as-is per Phase-0/1 scope; rename in Phase 5 along with the training-notebook path constant.
