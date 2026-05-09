# GPS-Denied Integrity Audit
**Pipeline 3 — All_In_One_Pipeline**
*Source-grounded audit, read-only. Performed: 2026-05-09. Source state: tip of `Lean_Online_Pipeline` branch.*

This audit traces every site in the runtime where simulator/GPS-truth latitude/longitude could influence the estimator or visual search. The thesis claim depends on having only one such site (the initialisation prior at $t_0$); any other site that reaches `update_position` represents a recurring GPS dependence and must be reported.

---

## Classification rubric

Each site is labelled with one of:

- **A — allowed initialisation prior**: used once at startup to set the EKF / local-NED origin.
- **B — allowed benchmarking/logging**: written to results.csv or analysis_extras.csv; never feeds back into the estimator or visual search.
- **C — questionable fallback correction**: feeds into the estimator under a defined operational condition (e.g. visual gate failure), motivated by a pragmatic safety net.
- **D — invalid leakage into estimator**: feeds into the estimator unconditionally and breaks the GPS-denied claim.
- **E — unclear, needs manual confirmation**.

---

## Summary table

| # | Site | File:lines | Class | One-line summary |
|---|---|---|---|---|
| GP1 | Live mode bootstrap | `runtime/run_pipeline.py:625-639` | **A** | First valid SimConnect GPS sample sets `lat0/lon0/alt0/heading0`. |
| GP2 | File mode bootstrap | `runtime/run_pipeline.py:288-305` | **A** | Row-0 GPS sets `lat0/lon0/alt0/heading0`. |
| GP3 | Per-frame IMU dict (live) | `runtime/run_pipeline.py:680-706` | **B** | `imu_data["lat"]/["lon"]` come from `ekf_state`, not `row`. |
| GP4 | Per-frame IMU dict (file) | `runtime/run_pipeline.py:316-338` | **B** | Same — EKF state, not GPS. |
| GP5 | Cold-start search centre | `src/best_first_search.py` (called from `src/temporal_searcher.py:121`) | **B** | Search centre = EKF position. |
| GP6 | Temporal-mode search centre | `src/temporal_searcher.py:296` | **B** | Same — EKF position. |
| GP7 | Visual position update | `src/ekf_ins.py:398-446` (called from `runtime/run_pipeline.py:386` file, `:748` live) | **B** | Measurement is the homography lat/lon. |
| GP8 | **Live-mode sim-GPS fallback** | `runtime/run_pipeline.py:752-763` | **C / D** | **Sim GPS fed into EKF when `gate_pass == False`. R = 200²= 40 000 m². NOT in file mode.** |
| GP9 | results.csv `gps_lat`/`gps_lon` | `runtime/run_pipeline.py:67, 409-410, 785-786` | **B** | Logging only. |
| GP10 | analysis_extras.csv homography offsets | `runtime/run_pipeline.py:528-538, 847-855` | **B** | Diagnostic, not fed back. |
| GP11 | Search-radius floor | `src/temporal_searcher.py:296-300`; `src/particle_filter.py::get_search_region` | **B** | Constant floor `FIRST_PASS_SEARCH_RADIUS_M = 500 m`. |
| GP12 | Particle filter init (low-confidence cold start) | `src/temporal_searcher.py:206` | **B** | EKF-derived seed. |
| GP13 | IMU fallback path (no first-pass tiles) | `src/temporal_searcher.py:569` | **B** | EKF-derived. |
| GP14 | EKF batch CSV preprocessor | `src/ekf_ins.py:573-672` (`_run_ekf_core`); `_check_gps_leakage:707-727` | **A** + leak check | Row-0 only; built-in leak detector warns on bit-identical reuse. |

---

## Verdict

| Mode | Verdict | Why |
|---|---|---|
| **File replay (`--source file`)** | ✅ **GPS-denied after row-0 initialisation.** | Only GP2 (init prior) + GP4/GP6/GP7 (EKF-derived search + visual update) + GP9/GP10 (logging). No sim-truth feedback path. |
| **Live SimConnect (`--source simconnect`)** | ⚠️ **NOT fully GPS-denied.** GP8 feeds sim GPS into the EKF whenever the visual quality gate fails, at R = 200² = 40 000 m². On the reference run `live_020_Odense_f1` (125 frames, 120 gate passes), **5 of 125 frames received a sim-GPS Kalman update** (≈ 4%). | Live mode contains a GP8 fallback that file mode does not. |

The fallback is annotated in source as an operational safety net (lines 753-757 of `runtime/run_pipeline.py`). Its intent is benign — it keeps the search region anchored inside the reference map when visual fails for several consecutive frames. Its scientific consequence is what this audit names: live-mode results that include any not-gate-pass frame are **not** purely GPS-denied; they include a soft GPS anchor with σ = 200 m.

---

## Detailed findings

### GP1 — Live mode bootstrap (Class A)
```python
# runtime/run_pipeline.py:625-639
print("[run_pipeline] Waiting for first valid SimConnect sample...")
while True:
    row = source.get_latest_row()
    if row and row.get("latitude") and abs(row["latitude"]) > 1.0:
        break
    time.sleep(0.05)

lat0     = row["latitude"]
lon0     = row["longitude"]
alt0     = barometric_altitude(row.get("barometer_pressure") or 1013.25)
heading0 = math.degrees(row.get("heading_magnetic") or 0.0)
mag_dec_deg, mag_inc_deg = get_mag_field(lat0, lon0, alt0)
ekf      = ErrorStateEKF(lat0, lon0, alt0, heading0, None,
                         mag_dec_deg=mag_dec_deg, mag_inc_deg=mag_inc_deg)
```
**Class A.** The first valid GPS sample sets the local-NED reference origin. After this point, position is read out as `lat = lat0 + pos_n / R_E` (see `ekf_ins.py:448-454`); subsequent state propagation never reads `row["latitude"]` again *except* via GP8.

### GP2 — File mode bootstrap (Class A)
```python
# runtime/run_pipeline.py:288-305
def _init_ekf(raw_df: pd.DataFrame, start_row: int):
    lat0      = raw_df["latitude"].iloc[0]
    lon0      = raw_df["longitude"].iloc[0]
    alt0      = barometric_altitude(raw_df["barometer_pressure"].iloc[0])
    heading0  = np.degrees(raw_df["heading_magnetic"].iloc[0])
    ...
    ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0, ...)
    prev_ts = None
    for i in range(start_row + 1):
        row_dict = raw_df.iloc[i].to_dict()
        step_ekf(ekf, row_dict, prev_ts)
        prev_ts = row_dict["timestamp"]
    return ekf, prev_ts
```
**Class A.** Identical pattern to GP1. The warm-up loop (lines 301-304) consumes IMU data only — `step_ekf` does not read `row["latitude"]`/`row["longitude"]`.

### GP3 — Per-frame IMU dict (live mode) (Class B clean)
```python
# runtime/run_pipeline.py:688-706
ekf_state = ekf.get_state()
ekf_lat   = ekf_state["latitude"]
ekf_lon   = ekf_state["longitude"]
ekf_yaw   = ekf_state["yaw"]
...
imu_data = {
    "lat":           ekf_lat,
    "lon":           ekf_lon,
    "heading":       ekf_yaw,
    "pos_sigma":     math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
    ...
}
```
**Class B clean.** `imu_data["lat"]/["lon"]` are EKF-derived. The variable `row["latitude"]` is *not* read here.

### GP4 — Per-frame IMU dict (file mode) (Class B clean)
```python
# runtime/run_pipeline.py:316-338  (_process_one_frame)
ekf_state = ekf.get_state()
ekf_lat   = ekf_state["latitude"]
...
imu_data = {
    "lat":           ekf_lat,
    "lon":           ekf_lon,
    ...
}
```
Same pattern as GP3.

### GP5, GP6 — Search centres (Class B clean)
- `src/temporal_searcher.py:121` (cold start): `searcher.search(query, imu_data["lat"], imu_data["lon"])`.
- `src/temporal_searcher.py:296`: `center_lat, center_lon = imu_data["lat"], imu_data["lon"]`; passed to `MetaTileBuilder.run(imu_lat=…, imu_lon=…)`.

In both cases, the search centre is the EKF estimate, not the simulator/GPS truth. Visual search is GPS-independent after $t_0$.

### GP7 — Visual position update (Class B clean)
```python
# src/ekf_ins.py:398-446
def update_position(self, lat_meas, lon_meas, R_pos_m2=None):
    """Visual position measurement update.
    Converts measured lat/lon → NED, computes innovation against
    current pos_n/pos_e, and runs a standard Kalman update on error
    states [8:10]."""
    ...
```
The function is **measurement-source agnostic**: it does not know whether `lat_meas/lon_meas` came from visual or from GPS. The classification of any individual call depends on its caller. Callers in this audit:
- `runtime/run_pipeline.py:386` (file mode): caller passes `homo_pos` (Class B clean).
- `runtime/run_pipeline.py:748` (live, gate_pass): caller passes `homo_pos` (Class B clean).
- `runtime/run_pipeline.py:762` (live, not gate_pass): caller passes `sim_lat, sim_lon` ← **GP8** (Class C/D).

### GP8 — Live-mode sim-GPS fallback (Class C / Class D)
```python
# runtime/run_pipeline.py:752-763
if not gate_pass:
    # Live mode: SimConnect GPS is available even when visual
    # localization fails.  Feed it as a very loose anchor
    # (R = 200² = 40 000 m²) so the EKF search region stays
    # inside the reference map.  Visual updates (R = 900–3 600 m²)
    # still dominate whenever they occur.
    sim_lat = row.get("latitude") if row else None
    sim_lon = row.get("longitude") if row else None
    if (sim_lat is not None and sim_lon is not None
            and abs(float(sim_lat)) > 1.0):
        ekf.update_position(float(sim_lat), float(sim_lon),
                            R_pos_m2=200.0 ** 2)
```
**Class C / Class D depending on framing.**

- *As shipped*, GP8 is **Class C** (questionable fallback correction): conditional, motivated, with bounded measurement noise. It does not unconditionally inject GPS, and R = 200 m is loose enough that it acts as a soft anchor rather than a precise fix.
- *In the absence of explicit thesis disclosure*, GP8 is effectively **Class D** (invalid leakage): a reader of the live-mode results would not know that 4–10 % of frames received a soft GPS pull.

**This site is the entire reason the live-mode aggregate metric is not a pure GPS-denied result.**

`run_file_mode` does **not** contain this branch (compare `_process_one_frame:374-386` — only the gate-pass visual update is performed; if the visual gate fails, the EKF coasts on IMU dead reckoning alone).

### GP9 — `gps_lat`/`gps_lon` columns in results.csv (Class B logging-only)
```python
# runtime/run_pipeline.py:67  (RESULT_COLUMNS)
"gps_lat", "gps_lon", "gps_alt_m",

# runtime/run_pipeline.py:409-410  (file mode)
"gps_lat":            row_dict.get("latitude"),
"gps_lon":            row_dict.get("longitude"),

# runtime/run_pipeline.py:785-786  (live mode)
"gps_lat":            row.get("latitude") if row else None,
"gps_lon":            row.get("longitude") if row else None,
```
**Class B logging-only.** Written to CSV for offline evaluation (e.g. haversine error vs. GT in `live_analysis.ipynb`). The runtime never reads these back.

### GP10 — `analysis_extras.csv` homography offsets (Class B logging-only)
File mode (`runtime/run_pipeline.py:528-538`) and live mode (`runtime/run_pipeline.py:847-855`) both compute:
```python
off_n = round((homo_corrected_lat - gps_lat) * 111320.0, 1)
off_e = round((homo_corrected_lon - gps_lon) * 111320.0
              * math.cos(math.radians(gps_lat)), 1)
```
and write to `analysis_extras.csv` (only when `SAVE_ANALYSIS_DATA = True`). Diagnostic only; not read by the runtime.

### GP11 — Search-radius floor (Class B clean)
```python
# src/temporal_searcher.py:296-300
center_lat, center_lon = imu_data["lat"], imu_data["lon"]
search_radius_m = max(
    region["radius_tiles"] * self.cfg.TILE_SIZE_METERS,
    self.cfg.FIRST_PASS_SEARCH_RADIUS_M,
)
```
Floor is a constant (`FIRST_PASS_SEARCH_RADIUS_M = 500 m`); no GPS dependence.

### GP12 — Particle filter init (low-confidence cold start) (Class B clean)
```python
# src/temporal_searcher.py:196-208
if gate_pass:
    init_lat, init_lon = homo_position
    spread = self.cfg.PARTICLE_INIT_SPREAD_HIGH_CONF
elif score >= 100 and position:
    init_lat, init_lon = position
    spread = self.cfg.PARTICLE_INIT_SPREAD_MED_CONF
else:
    init_lat, init_lon = imu_data["lat"], imu_data["lon"]   # ← EKF-derived
    spread = self.cfg.PARTICLE_INIT_SPREAD_LOW_CONF
```
EKF-derived seed when no visual hypothesis is strong enough.

### GP13 — IMU fallback path (Class B clean)
```python
# src/temporal_searcher.py:563-572
if self.particle_filter is not None:
    est_x, est_y, est_hdg = self.particle_filter.get_estimate()
    fb_lat, fb_lon = tile_to_latlon(est_x, est_y, self.cfg.TMS_ZOOM_LEVEL)
    ...
else:
    fb_lat, fb_lon = imu_data["lat"], imu_data["lon"]   # ← EKF-derived
    fb_heading = imu_data["heading"]
return {
    "position": (fb_lat, fb_lon),
    ...
}
```
EKF-derived position used when there are no first-pass tiles and the particle filter is missing. Caller does not pass this through `update_position`.

### GP14 — EKF batch CSV preprocessor (Class A with explicit leak detector)
```python
# src/ekf_ins.py:573-672  (_run_ekf_core)
# Initialise from row 0 (GPS for reference origin ONLY)
lat0 = df['latitude'].iloc[0]
lon0 = df['longitude'].iloc[0]
...
for i in range(len(df)):
    # only IMU/baro/mag/airspeed updates; GPS never re-read
    ...

# src/ekf_ins.py:707-727  (_check_gps_leakage)
def _check_gps_leakage(est_df, raw_df, n_check=20):
    """Warn loudly if estimated lat/lon are bit-for-bit identical to raw GPS."""
    ...
    if np.array_equal(est_lat, raw_lat) and np.array_equal(est_lon, raw_lon):
        warnings.warn("GPS LEAKAGE DETECTED: ...", RuntimeWarning, ...)
```
**Class A with built-in leak detector.** This is the offline batch wrapper used by analysis tooling; it explicitly warns if the produced estimate sequence ever becomes bit-identical to the raw GPS sequence. Confirms the design intent.

---

## Notes on file-mode replay availability

The repository ships with one recorded log: `Logs_Run_20260321_162024/` (frames + IMU CSV). This log was recorded over Vejle/Copenhagen-region terrain. The current active reference map (per `config/config.py:23-24, 37, 66-69`, updated 2026-05-09) is **Odense** (`REFERENCE_MAP_ODENSE/`). The recorded log GPS coordinates are **outside** the Odense tile bounds (`TILE_X 34630-34689, TILE_Y 44916-44948`).

**Consequence:** running `python -m runtime.run_pipeline --source file --run-id smoke_phase0_file` against the current configuration would fail at cold-start (no tiles within search radius of the EKF-initialised position), or run with method=`imu_fallback` for every frame.

**File-mode classification: BROKEN AS CONFIGURED — code is sound, configuration mismatch.** A clean file-mode evaluation requires either (a) a recorded log over the Odense map, or (b) re-pointing config to the Vejle map for which the existing log has matching tiles, or (c) recording a new log.

This is documented for transparency, not as a code defect.

---

## Recommendations for thesis evaluation

1. **For headline GPS-denied results**, run file-mode replay against a matched reference map (record an Odense flight, or restore the Vejle map as the active config for a single evaluation). File-mode is the only path that is purely GPS-denied as shipped.
2. **For live-mode results**, either:
   - (a) Disable GP8 (comment out `runtime/run_pipeline.py:752-763`) and re-run for the thesis evaluation flight; or
   - (b) Report aggregate metrics partitioned by `gate_pass` and exclude not-gate-pass frames from the GPS-denied aggregate; or
   - (c) Frame GP8 explicitly as an operational safety mechanism, label it in diagrams as auxiliary, and report two metrics: "method-only" (gate-pass frames) and "operational" (all frames).
3. **In all cases**, the thesis text must use the precise framing **"GPS-denied after initial geodetic prior at $t_0$"** rather than "GPS-free" or "never uses GPS". The system uses GPS exactly once, by design, to set the local-NED origin.

See `BS_CHECK.md` for the brutally honest assessment of whether the current claim is defensible.
