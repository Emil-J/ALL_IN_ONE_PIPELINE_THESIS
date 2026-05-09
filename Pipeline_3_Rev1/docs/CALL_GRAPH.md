# Live-Mode Runtime Call Graph
**Pipeline 3 — All_In_One_Pipeline**
*Source-grounded, line-numbered. Verified against tip of `Lean_Online_Pipeline` branch on 2026-05-09.*

This document traces every function call on the live runtime hot path of `python -m runtime.run_pipeline --source simconnect`. Every block points at file:line so a regenerated diagram can re-verify the structure.

For the brief Mermaid view: [`Diagrams/03_live_runtime_pipeline.mmd`](Diagrams/03_live_runtime_pipeline.mmd) and [`Diagrams/05_ekf_visual_fusion.mmd`](Diagrams/05_ekf_visual_fusion.mmd).

For the file-mode equivalent: see [`Diagrams/04_file_replay_pipeline.mmd`](Diagrams/04_file_replay_pipeline.mmd) (currently broken-as-configured — see [`CURRENT_BEHAVIOUR_BASELINE.md`](CURRENT_BEHAVIOUR_BASELINE.md)).

---

## 1. Top-level dispatch

```
$ python -m runtime.run_pipeline --source simconnect --run-id <id>
                                                                     ┌───────── runtime/run_pipeline.py ─────────┐
runtime/run_pipeline.py::main             (line 945)                 │ argparse → mode dispatch                  │
   ├─ _parse_args()                       (line 231)                 └───────────────────────────────────────────┘
   ├─ config.ensure_output_dirs()         (config/config.py:254)
   ├─ _set_deployment_flags(args.debug)   (line 262)
   ├─ run_dir, run_id = _build_run_dir(args)  (line 251)
   └─ run_simconnect_mode(args, run_dir, run_id)  (line 608)
```

---

## 2. Initialisation phase

```
run_simconnect_mode  (run_pipeline.py:608)
   │
   ├─ _init_models()                                 (run_pipeline.py:267)
   │     ├─ load_semantic_model(SEM_PATH, "cuda")    (src/semantic_model.py)
   │     │       reads SemanticTerrainSegmentationModel/best.pth
   │     ├─ initialize_matcher("cuda", 2048)         (src/geometric_matcher.py)
   │     │       loads SuperPoint + LightGlue
   │     ├─ TileLoader(REFERENCE_TILES_DIR,
   │     │            REFERENCE_PRED_DIR, …)         (src/tile_utils.py)
   │     │       indexes REFERENCE_MAP_ODENSE/{aerial,prediction}/16/{x}/{y}.png
   │     └─ FeatureStoreLoader(REFERENCE_FEATURES_PATH,
   │                            device="cuda")       (Dataset_Preprocessing/feature_store.py)
   │             opens REFERENCE_MAP_ODENSE/reference_features.h5
   │             ⚠ imported via sys.path.insert     (run_pipeline.py:36-38) — see CODEMAP § 10
   │
   ├─ TemporalSearcher(model, matcher, tiles, cfg, feature_store=fs)
   │                                                 (src/temporal_searcher.py:46)
   │     constructs MetaTileBuilder + SemanticConfirmer
   │
   ├─ source = SimConnectLiveSource(); source.connect()
   │                                                 (runtime/simconnect_adapter.py)
   │     starts background SimConnect thread; opens screen-capture region
   │
   ├─ Bootstrap loop                                  (run_pipeline.py:625)
   │     wait until row.get("latitude") is non-trivial
   │
   ├─ get_mag_field(lat0, lon0, alt0)                (src/wmm_declination.py)
   │     reads WMM2025COF/WMM2025.COF
   │
   └─ ekf = ErrorStateEKF(lat0, lon0, alt0, hdg0, …, mag_dec, mag_inc)
                                                     (src/ekf_ins.py:115-211)
         initialises 10D state vector + 10×10 covariance
         **GP1: allowed initialisation prior — single GPS sample consumed here**
```

---

## 3. Per-frame main loop

```
MAIN LOOP (run_pipeline.py:667-905)

  ┌─ Section 3a — IMU step ─────────────────────────────────────────────────────┐
  │  row = source.get_latest_row()                                              │
  │  if row.timestamp != last_imu_ts:                                           │
  │      step_ekf(ekf, row, prev_ts)                  (src/ekf_ins.py:502)      │
  │          ├─ MSFS axis remap (inline)                                        │
  │          ├─ ekf.predict(omega, accel, dt)         (ekf_ins.py:213-247)      │
  │          ├─ ekf.update_barometer(h, ts)           (ekf_ins.py:311-317)      │
  │          ├─ ekf.update_accel_mag(a, hdg)          (ekf_ins.py:249-309)      │
  │          └─ ekf.update_airspeed(V_air, hdg)       (ekf_ins.py:319-396)      │
  │      prev_ts = row.timestamp                                                │
  │      last_imu_ts = row.timestamp  ← prevents repeated step on same row      │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3b — Wait for new frame ───────────────────────────────────────────┐
  │  frame_img, frame_id, frame_capture_ts = source.get_latest_frame()          │
  │  if frame_img is None or frame_id == last_frame_id:                         │
  │      time.sleep(0.005); continue                                            │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3c — Build IMU dict for visual search ─────────────────────────────┐
  │  ekf_state = ekf.get_state()                                                │
  │  imu_data = {                                                               │
  │      "lat":           ekf_state["latitude"],   # GP3 — EKF-derived (clean)  │
  │      "lon":           ekf_state["longitude"],                               │
  │      "heading":       ekf_state["yaw"],                                     │
  │      "pos_sigma":     sqrt(max(ekf.P[8,8], ekf.P[9,9])),                    │
  │      "velocity_mps":  …,                                                    │
  │      "gyro_z_dps":    row["gyro_z"]·180/π,                                  │
  │      "pitch":         row["pitch"], "roll": row["bank"],                    │
  │  }                                                                          │
  │  ts = row.get("timestamp", time.time())                                     │
  │  ekf_state_before = ekf_state                  ← captured for trace JSON    │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3d — Visual processing ────────────────────────────────────────────┐
  │  result = searcher.process_frame(frame, imu_data, timestamp=ts)             │
  │                                  (src/temporal_searcher.py:70)              │
  │      → frame_count == 0:  _process_frame_0       (line 100)                 │
  │      → frame_count > 0:   _process_frame_N       (line 278)                 │
  │      [see Section 4 below for internals]                                    │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3e — Look-ahead correction ────────────────────────────────────────┐
  │  if homo_pos is not None:                       (run_pipeline.py:724-735)   │
  │      h_rad = math.radians(ekf_yaw)                                          │
  │      effective_lookahead = LOOKAHEAD_M · cos(bank_rad)    # LOOKAHEAD = 110m│
  │      corr_north = -effective_lookahead · cos(h_rad)                         │
  │      corr_east  = -effective_lookahead · sin(h_rad)                         │
  │      homo_corr_lat = homo_lat_raw + corr_north / 111320.0                   │
  │      homo_corr_lon = homo_lon_raw + corr_east  / (111320 · cos(lat))        │
  │      homo_pos = (homo_corr_lat, homo_corr_lon)                              │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3f — Adaptive measurement noise ───────────────────────────────────┐
  │  if homo_pos is not None:                       (run_pipeline.py:737-746)   │
  │      sc_method = result.get("method", "")                                   │
  │      if sc_method == "cold_start":                                          │
  │          r_used = R_COLD_START   # 100²                                     │
  │      else:                                                                  │
  │          r_used = R_HIGH if (cs > 0.5 and ni > 100) else R_MED  # 30²/60²   │
  │      if bank_rad > TURN_ROLL_THRESHOLD_RAD:        # ~20°                   │
  │          r_used *= TURN_R_MULTIPLIER    # 2.0                               │
  │      if not meta_verified:                                                  │
  │          r_used *= 2.0                                                      │
  │      r_used *= max(0.5, 2.0 - 1.5 · sem_conf)                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3g — EKF update (the fork) ────────────────────────────────────────┐
  │  if gate_pass and homo_pos is not None:         (run_pipeline.py:747-750)   │
  │      ekf.update_position(homo_corr_lat, homo_corr_lon, R=r_used)            │
  │              (src/ekf_ins.py:398-446)                  # GP7 — visual upd.  │
  │      gate_count += 1                                                        │
  │                                                                             │
  │  if not gate_pass:                              (run_pipeline.py:752-763)   │
  │      ───────────────────────────────────────────────────────────────────    │
  │      ⚠ GP8 — LIVE-MODE SIM-GPS FALLBACK (auxiliary, not part of method)     │
  │      ekf.update_position(sim_lat, sim_lon, R=200²)   # 40 000 m²            │
  │      ───────────────────────────────────────────────────────────────────    │
  │      [no equivalent in file mode — see _process_one_frame:374-386]          │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─ Section 3h — Final state read-out and write row ───────────────────────────┐
  │  final = ekf.get_state()                        (ekf_ins.py:448-474)        │
  │  pos_sigma = sqrt(max(ekf.P[8,8], ekf.P[9,9]))                              │
  │  result_row = { … 31 columns … }                (run_pipeline.py:772-805)   │
  │  writer.writerow(result_row)                                                │
  │                                                                             │
  │  Optional output writers (controlled by config flags):                      │
  │    SAVE_QUERY_FRAMES     → flight_data/frame_NNNN.jpg                       │
  │    SAVE_IMU_ROWS         → flight_data/frame_NNNN_imu.json                  │
  │    SAVE_ANALYSIS_DATA    → px4_gps_input.csv + analysis_extras.csv          │
  │    SAVE_TIMING_DATA      → timing_data.csv                                  │
  │    SAVE_PIPELINE_TRACE   → pipeline_data/frame_NNNN/{query.jpg, …}          │
  │  See FLAGS.md for the full table.                                           │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. `TemporalSearcher.process_frame` internals

The orchestrator dispatches by `frame_count`:

### 4.1 Frame 0 — cold start (`_process_frame_0`, lines 100-272)

```
_process_frame_0
  ├─ heading_deg = imu_data["heading"]                       ← GP3 (EKF-derived)
  ├─ rotation_angle = -heading_deg
  ├─ query_rotated, rot_M = rotate_image(frame, rotation_angle)
  │                              (src/visual_measurement.py)
  ├─ query_for_match = self._resize_rotated(query_rotated)   # ≤ 1280 px
  │
  ├─ searcher = BestFirstSearcher(matcher, tiles, cfg, feature_store)
  │                              (src/best_first_search.py)
  ├─ search_result = searcher.search(query_for_match,
  │                                   imu_data["lat"], imu_data["lon"])
  │                                                         ← GP5 (search centre = EKF)
  │      For each candidate tile within IMU_SEARCH_RADIUS_METERS:
  │        ├─ tile_img = TileLoader.load_aerial(tx, ty)
  │        ├─ matcher.match(query, tile)                    (SP+LG)
  │        └─ rank by inlier count
  │      Returns: best_tile, score, ranked_tiles, match_result
  │
  ├─ if search_result["match_result"] and score >= 4:
  │      mr = search_result["match_result"]
  │      src_pts, dst_pts = mr["keypoints1"][matches[:,0]], mr["keypoints2"][matches[:,1]]
  │      dual = compute_dual_homography(src_pts, dst_pts, qw, qh, RANSAC_THRESH)
  │                              (src/visual_measurement.py)
  │              MAGSAC++ vs LMEDS, winner by CShape × inliers
  │      measurements_dict = extract_visual_measurements(dual["winner_H"], …,
  │                                                       pitch_rad, roll_rad)
  │              5 methods: nadir_corrected → trimmed → inlier → weighted → projected
  │      cascade = self._build_cascade(pitch_rad, roll_rad)
  │      for mname in cascade:
  │          if measurements_dict[mname]["valid"]:
  │              homo_position = measurements_dict[mname]["latlon"]; break
  │
  ├─ Quality gate: gate_pass = (cshape > 0.3 and inliers > 20 and homo_position)
  ├─ Choose PF init position based on gate_pass / score
  │      gate_pass:        init_pos = homo_position;       spread = HIGH_CONF
  │      score ≥ 100:      init_pos = (search-best);       spread = MED_CONF
  │      else:             init_pos = imu_data["lat,lon"]; spread = LOW_CONF  ← GP12 clean
  │
  ├─ Initialise self.particle_filter = ParticleFilter(N=100, init_pos, init_hdg, spread)
  │                              (src/particle_filter.py)
  │
  ├─ Semantic Branch A (parallel-ish):
  │   query_processed = preprocess_query_frame(frame, 512×288 → 512×512)
  │                              (src/image_utils.py)
  │   query_semantic_map = semantic_model.predict(query_processed)
  │                              (src/semantic_model.py)
  │
  └─ return {position, heading, score, method="cold_start", visual_quality,
             gate_pass, homo_position, query_semantic_map, ranked_tiles, …}
```

### 4.2 Frame 1+ — temporal tracking (`_process_frame_N`, lines 278-523)

```
_process_frame_N
  ├─ Step 1  pf.predict(dt, velocity_mps, gyro_z_dps)      (particle_filter.py)
  ├─ Step 2  region = pf.get_search_region()
  │          center_lat, center_lon = imu_data["lat,lon"]    ← GP6 clean
  │          search_radius_m = max(region.radius·tile_size, 500m)   ← GP11 clean
  ├─ Step 3  query_rotated, _ = rotate_image(frame, -heading)
  │          query_for_match = _resize_rotated(query_rotated)
  ├─ Step 3b query_semantic_map = semantic_model.predict(preprocess_query_frame(frame))
  │
  ├─ Step 4  meta_result = MetaTileBuilder.run(query_for_match, center_lat, center_lon,
  │                          ts, search_radius_m, query_semantic_map)
  │                              (src/meta_tile_builder.py)
  │      ├─ first pass: find_tiles_within_radius() → SP+LG over each tile
  │      │      uses FeatureStoreLoader.has_tile/get_features fast path
  │      ├─ optional histogram pre-filter (SemanticTileScorer)
  │      ├─ second pass: 8-neighbour expansion of top-1
  │      ├─ stitch top-3 → meta_tile
  │      └─ verify: SP+LG(query, meta_tile) ≥ METATILE_MATCH_THRESHOLD (25)
  │
  ├─ if meta_result is None:                              (line 348)
  │      → _imu_fallback_result(...)                      (line 558)
  │            uses pf.get_estimate() OR imu_data         ← GP13 clean
  │            method="imu_fallback"
  │
  ├─ Step 5  dual = compute_dual_homography(...)
  │          measurements_dict = extract_visual_measurements(...)
  │          cascade selects best valid measurement → homo_position, homo_tile_pos
  │
  ├─ Step 6  measurements = […tile centres or homo sub-tile pos with score…]
  │          pf.update(measurements); pf.resample()
  │
  ├─ Step 7  confirm_result = SemanticConfirmer.confirm(query_semantic_map,
  │                                                      meta_result["meta_tile"], …)
  │                              (src/semantic_confirmer.py)
  │
  ├─ Step 8  est_x, est_y, est_hdg = pf.get_estimate()
  │          est_lat, est_lon = tile_to_latlon(est_x, est_y, zoom)
  │
  ├─ Step 9  Quality-gated final_position selection:
  │          if gate_pass: final_position = homo_position
  │          else:         final_position = pf_pos
  │
  ├─ Step 10 if pf.check_divergence(): self.frame_count = 0   ← cold restart on next frame
  │
  └─ return {position, heading, method="temporal_tracking", visual_quality,
             gate_pass, homo_position, particle_spread, n_eff, semantic_confidence,
             meta_tile_verified, ranked_tiles, query_semantic_map, …}
```

---

## 5. EKF internals (`src/ekf_ins.py`)

```
ErrorStateEKF                                       (lines 115-474)
  state vector: [δθ (3), δb_g (3), δw (2), δp (2)] = 10D error
  nominal:      q_tilde (quaternion), pos_n, pos_e, pos_d, vel_n, vel_e, vel_d, gyro_bias, wind_n, wind_e
  P:            10×10 covariance
  origin:       lat0_rad, lon0_rad, alt0  (GP1/GP2)
  output:       get_state() returns lat = lat0 + pos_n/R_E, lon = lon0 + pos_e/(R_E·cos(lat0)), etc.

  predict(omega, accel, dt)                         (lines 213-247)
    omega_corrected = omega - gyro_bias
    dq = expq(dt/2 · omega_corrected); q_tilde = normalise(q_tilde · dq)
    F[0:3,3:6] = 0.5·dt·R_nb;  F[8:10,8:10] = I
    P = F·P·F.T + Q                                  (Q includes pos process noise)

  update_barometer(altitude_m, ts)                  (lines 311-317)
  update_accel_mag(accel, mag_heading_deg)          (lines 249-309)
       accel update: roll/pitch via gravity vector (yaw column zeroed)
       mag update:   scalar heading innovation (angle-wrapped)
  update_airspeed(V_air, mag_heading_deg)           (lines 319-396)
       observes wind_n, wind_e
       blends velocity per α (manoeuvre/wind-converged state)

  update_position(lat_meas, lon_meas, R_pos_m2)     (lines 398-446)
       z = (lat-lat0)·R_E,  (lon-lon0)·R_E·cos(lat0)
       innovation = z - (pos_n, pos_e)
       H[0,8]=H[1,9]=1; standard Kalman update
       δstate applied to all blocks (orientation, bias, wind, position)

  step_ekf(ekf, row, prev_ts)                       (lines 502-570)
       MSFS axis remap (ft/s² → m/s², handedness, gravity synthesis)
       calls predict + update_barometer + update_accel_mag + update_airspeed
```

---

## 6. Optional output writers

Hot-path side-effects controlled by `config.py` flags. All default to `False` except `SAVE_PIPELINE_TRACE`. See [`FLAGS.md`](FLAGS.md) for full timing impact and combinations.

| Flag | Code site | Output |
|---|---|---|
| `SAVE_QUERY_FRAMES` | `run_pipeline.py:809-813` | `flight_data/frame_NNNN.jpg` |
| `SAVE_IMU_ROWS` | `run_pipeline.py:816-822` | `flight_data/frame_NNNN_imu.json` |
| `SAVE_ANALYSIS_DATA` | `run_pipeline.py:824-863` | `px4_gps_input.csv` + `analysis_extras.csv` |
| `SAVE_TIMING_DATA` | `run_pipeline.py:866-880` + `temporal_searcher.py` | `timing_data.csv` |
| `SAVE_PIPELINE_TRACE` | `run_pipeline.py:883-896` (live), `:548-560` (file) | `pipeline_data/frame_NNNN/{query.jpg, query_rotated.jpg, semantic_mask.png, reference_tile.png, matches.png, imu.json, trace.json}` |

---

## 7. Live vs. file mode delta

| Stage | Live (`run_simconnect_mode`) | File (`run_file_mode` → `_process_one_frame`) |
|---|---|---|
| Init | GP1 (line 625) — first valid SimConnect row | GP2 (line 288) — `raw_df["latitude"].iloc[0]` |
| IMU step | inline, gated on `last_imu_ts` (line 672) | inline in `_process_one_frame` (line 318) |
| Visual update on gate-pass | line 748 — `ekf.update_position(homo, R=adaptive)` | line 386 — same |
| Visual update on gate-fail | **GP8** (lines 752-763) — `ekf.update_position(sim, R=200²)` | **— absent —** |
| Frame source | `SimConnectLiveSource.get_latest_frame()` (screen capture) | `load_image(frame_path)` (JPEG read) |
| `inference_ms` reported | yes (capture-ts → estimate-ts) | always `None` |

---

## 8. Verification

To re-verify the line numbers above, run:

```powershell
# from repo root
$ROOT = "Pipeline_3_Rev1"
& "C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\.final_Pipeline_venv\Scripts\python.exe" `
  -c "import re, pathlib; src = pathlib.Path('$ROOT/runtime/run_pipeline.py').read_text(); print('GP8 at line', src[:src.index('if not gate_pass:\\n                    # Live mode')].count('\\n')+1)"
```

A future Claude session generating Mermaid diagrams should re-grep `run_pipeline.py`, `temporal_searcher.py`, and `ekf_ins.py` for the function names listed in this document and update the Mermaid header comments if any line numbers have shifted.
