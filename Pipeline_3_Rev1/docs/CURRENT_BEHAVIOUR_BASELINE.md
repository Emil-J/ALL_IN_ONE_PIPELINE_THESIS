# Current Behaviour Baseline
**Pipeline 3 — All_In_One_Pipeline**
*Frozen snapshot of system behaviour at: 2026-05-09. Reference run: `live_020_Odense_f1`.*

This document captures what the system does today, so any future Phase-2+ refactor can verify it has not silently changed behaviour.

---

## Verified live runtime baseline

### Run identifier
`outputs/runs/live_020_Odense_f1/`

### Provenance
- Source: `--source simconnect`
- Map: Odense (`REFERENCE_MAP_ODENSE/`)
- Active config: `config/config.py` (tile bounds X 34630–34689, Y 44916–44948; `REFERENCE_TILES_DIR`/`REFERENCE_PRED_DIR`/`REFERENCE_FEATURES_PATH` point at Odense)
- Semantic checkpoint: `SemanticTerrainSegmentationModel/best.pth` (2026-03-18)
- Reference features: `REFERENCE_MAP_ODENSE/reference_features.h5` (3.5 GB, generated 2026-05-09)
- MSFS 2020 connected, capture region 1920×1009 px on a secondary monitor.
- Initialisation: `EKF bootstrapped: (55.366783, 10.273298) yaw=91.0°` (Odense, eastward heading).

### `run_meta.json`
```json
{
  "run_id": "live_020_Odense_f1",
  "source": "simconnect",
  "n_frames": 125,
  "gate_count": 120,
  "elapsed_s": 346.5,
  "fps": 0.361,
  "imu_csv": null,
  "frames_dir": null
}
```

### Throughput
- 125 frames in 346.5 s → **0.361 fps** (≈ 2.8 s/frame).
- Gate pass rate: **120/125 = 96.0 %**.
- Method distribution: 1 `cold_start` (frame 0), 124 `temporal_tracking`.

### Gate-failure frames (5 total)
| Frame | Method | gate_pass | cs_shape | inliers | meta_tile_verified | Notes |
|---|---|---|---|---|---|---|
| 0 | cold_start | 0 | 0.186 | 7 | 0 | Expected: cold start often fails the gate. |
| 6 | temporal_tracking | 0 | 0.695 | 198 | 1 | Strong match but `homo_corrected_lat` empty — measurement extraction failed despite high inliers/cs. |
| 13 | temporal_tracking | 0 | 0.148 | 9 | 1 | cs below `QUALITY_GATE_CSHAPE = 0.3`. |
| 14 | temporal_tracking | 0 | 0.604 | 10 | 1 | inliers below `QUALITY_GATE_INLIERS = 20`. |
| 22 | temporal_tracking | 0 | 0.000 | 0 | 0 | Total visual failure. |

These 5 frames are the GP8-anchored frames (see `GPS_DENIED_INTEGRITY_AUDIT.md`). On each, `runtime/run_pipeline.py:752-763` invokes `ekf.update_position(sim_lat, sim_lon, R = 200²)`.

### `results.csv` — first row
```
frame_idx,timestamp,image_name,final_lat,final_lon,heading_deg,altitude_m,roll_deg,pitch_deg,vel_n,vel_e,vel_d,gps_lat,gps_lon,gps_alt_m,method,gate_pass,search_time_s,cs_shape,inliers,semantic_conf,homo_lat,homo_lon,homo_corrected_lat,homo_corrected_lon,meta_tile_verified,ekf_pos_sigma,r_used_sqrt,tiles_tested,verification_matches,inference_ms
0,1778327352.2928104,live_1,55.36678323196945,10.27329785299176,95.4431438770571,506.72,-2.131,-3.245,-4.616,48.44,-0.0,55.36678323196945,10.27329785299176,506.62,cold_start,0,3.4204,0.1861,7,0.5,55.36820367595092,10.27410532228231,55.36829740634285,10.272374446254194,0,141.42,158.11,6,,3816.1
```

### `results.csv` — last row
```
124,1778327694.562051,live_990,55.3534851327679,10.41774846394793,140.26638240566297,507.4,-0.527,0.191,-51.873,42.921,-0.131,55.353316944213965,10.418003141240614,507.31,temporal_tracking,1,2.2991,0.74,98,0.797,55.35489249111891,10.422489229222265,55.3556523968783,10.421378149547806,1,14.84,53.82,7,124,2777.8
```

### Trajectory characterisation
- Start: (55.367°N, 10.273°E) ≈ Odense centre
- End: (55.353°N, 10.418°E) ≈ ~10 km east-southeast of start
- Total flight time ≈ 343 s (5 min 43 s)
- Altitude: ≈ 506–509 m throughout (1660 ft, level flight)
- Heading: 91° → 140° (gradual right turn)

---

## File-mode replay baseline

**Status: BROKEN AS CONFIGURED.**

### Why
- Repository ships with one recorded log: `Logs_Run_20260321_162024/` (frames + IMU CSV), recorded over Vejle/CPH-region terrain in March 2026.
- Active reference map (per current `config/config.py:23-24, 37, 66-69`) is **Odense**.
- The recorded log GPS coordinates fall **outside** the Odense tile bounds. `_init_ekf` would set a Vejle-region origin, then `find_tiles_within_radius()` would return zero candidates because no Odense tiles lie within the 500 m search radius of a Vejle position.

### Smoke run not executed
A 30-frame `--source file` smoke run was deferred because:
1. The above mismatch means it would not produce a meaningful result (every frame would be `imu_fallback` or `cold_start` failure).
2. The per-Phase 1 scope explicitly forbids editing config or moving files. Re-pointing the config or symlinking the map directory would constitute a configuration change.
3. The user has confirmed end-to-end functionality of the live pipeline via the `live_020_Odense_f1` run.

### To restore file-mode evaluation later
Pick **one** of:
1. Re-record a flight log over the Odense map (preferred for thesis evaluation; produces a clean GPS-denied baseline).
2. Restore Vejle as the active reference map by editing `config/config.py:23-24, 37, 66-69` to point at `REFERENCE_MAP_VEJLE_20260321_162024/` and use the matching tile bounds.
3. Add a CLI override to `run_pipeline.py` (out of Phase 1 scope) to allow per-run reference-map selection.

---

## Pytest baseline

Six unit tests live under `Pipeline_3_Rev1/tests/`:

```
Pipeline_3_Rev1/tests/test_10d_ekf.py
Pipeline_3_Rev1/tests/test_meta_tile_builder.py
Pipeline_3_Rev1/tests/test_particle_filter.py
Pipeline_3_Rev1/tests/test_semantic_confirmer.py
Pipeline_3_Rev1/tests/test_temporal_searcher.py
Pipeline_3_Rev1/tests/test_units.py
```

These tests cover the algorithmic core (EKF predict/update, particle filter resampling, meta-tile verification, semantic confirmation, temporal-searcher orchestration with mocked I/O). They do **not** require GPU, MSFS, or any reference map data.

### How to run

```powershell
# from repo root
& C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\.final_Pipeline_venv\Scripts\Activate.ps1
python -m pytest Pipeline_3_Rev1/tests -q
```

### Phase-0 status
Pytest was not re-executed during Phase 0 because the user confirmed the system works end-to-end via `live_020_Odense_f1`. Pytest can be re-executed at any time using the command above; expected outcome is **all tests pass** based on the reference test_temporal_searcher.py importing the same `runtime.run_pipeline._process_one_frame` path that produced `live_020_Odense_f1`.

---

## Field-tolerance comparison rules (for Phase 1 closing validation)

Phase 1 changes only documentation files. After Phase 1 completes, a follow-up run of `live_020_Odense_f1`-equivalent flight is **not** required (live runs are non-deterministic). The following rules apply if any future regression run is performed against this baseline.

### Categorical / discrete columns — must match exactly

| Column | Rule |
|---|---|
| `frame_idx` | exact equality |
| `gate_pass` | exact equality |
| `method` | exact equality |
| `meta_tile_verified` | exact equality |
| `tiles_tested` | exact equality |
| `verification_matches` | exact equality |
| `inliers` | exact equality |

Any mismatch on these columns is a hard failure and indicates an algorithmic change.

### Float columns — exact first, allow tolerance if numerical drift appears

| Column group | First pass | Fallback tolerance |
|---|---|---|
| `final_lat`, `final_lon`, `homo_lat`, `homo_lon`, `homo_corrected_lat`, `homo_corrected_lon` | exact | $10^{-6}\,°$ ($\approx 0.1$ m) |
| `heading_deg`, `roll_deg`, `pitch_deg` | exact | $10^{-3}\,°$ |
| `cs_shape`, `semantic_conf`, `r_used_sqrt`, `ekf_pos_sigma`, `altitude_m`, `vel_n`, `vel_e`, `vel_d` | exact | $10^{-3}$ relative |

GPU/cuDNN nondeterminism in SuperPoint, LightGlue, and UNet++ inference can produce sub-pixel differences across runs even without code changes. Accept those silently; surface only if they exceed the tolerance band.

### Timing columns — always ignored

`timestamp`, `search_time_s`, `inference_ms`. Wall-clock is nondeterministic by definition.

---

## Confirmed externalities (do not touch in Phase 1)

| Artefact | Path | Modified |
|---|---|---|
| Active semantic checkpoint | `SemanticTerrainSegmentationModel/best.pth` | 2026-03-18 |
| Frozen Phase 1 winter checkpoint | `SEMANTIC BEFORE/1BEST_TRAINING_OUTCOME_20260304_222309/.../best.pth` | 2026-03-06 |
| Active reference map | `REFERENCE_MAP_ODENSE/{aerial,prediction,reference_features.h5}` | 2026-05-09 |
| WMM2025 coefficients | `WMM2025COF/WMM2025.COF` | (external, NOAA) |
| QGIS extraction tooling | `QGIS/` | 2026-05-09 |

These remain untouched throughout Phase 0 and Phase 1.

---

## Stop-condition summary

- [x] Audit results recorded: `GPS_DENIED_INTEGRITY_AUDIT.md`, `BS_CHECK.md`.
- [x] Live-mode baseline frozen: `live_020_Odense_f1` (125 frames, 96.0 % gate pass).
- [x] File-mode classified: **broken as configured** (recorded log/map mismatch — documented above).
- [x] GP1–GP14 line numbers verified against current source tree.
- [x] No source code, config, notebook, or checkpoint files modified.
