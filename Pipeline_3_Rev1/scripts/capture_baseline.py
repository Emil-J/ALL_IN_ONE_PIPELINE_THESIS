"""
Baseline capture script — mirrors notebook Cells 1-4 exactly for 50 frames.
Writes per-frame table + summary to stdout AND outputs/baseline_50frames.txt.

Run from Pipeline_3_Rev1/:
    ..\\.final_Pipeline_venv\\Scripts\\python.exe scripts\\capture_baseline.py 2>&1
"""

import sys
import os
import copy
import time
import math
from pathlib import Path
from collections import Counter

# --- path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent           # Pipeline_3_Rev1/
REPO = ROOT.parent                 # All_In_One_Pipeline/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from config import config
from src.tile_utils import TileLoader, haversine_distance as _hav, tile_to_latlon
from src.image_utils import load_image
from src.geometric_matcher import initialize_matcher
from src.semantic_model import load_semantic_model
from src.temporal_searcher import TemporalSearcher
from src.ekf_ins import preprocess_imu_csv, ErrorStateEKF, step_ekf, barometric_altitude

# ── RUN PARAMETERS (match Cell 4 notebook) ──────────────────────────────────
RUN_N           = 50
START_ROW       = 0
LOOKAHEAD_M     = 110.0
R_HIGH          = 30.0 ** 2
R_MED           = 60.0 ** 2
TURN_ROLL_THRESHOLD_RAD = 0.26
TURN_R_MULTIPLIER       = 3.0

OUTPUT_FILE = ROOT / "outputs" / "baseline_50frames.txt"


def main():
    config.ensure_output_dirs()
    lines = []

    def tee(msg=""):
        print(msg)
        lines.append(msg)

    # ── Cell 2: Load data + EKF warmup ──────────────────────────────────────
    tee("=== BASELINE CAPTURE: 50 frames ===")
    tee(f"Config path: {ROOT}")

    imu_log = preprocess_imu_csv(config.IMU_CSV_PATH)
    tee(f"IMU log: {len(imu_log)} rows")

    frame_dir = config.QUERY_FRAMES_DIR
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    tee(f"Found {len(frame_files)} frames")

    frame_map = {}
    for fp in frame_files:
        ts_str = fp.stem.replace("frame_", "")
        try:
            frame_map[round(float(ts_str), 3)] = fp
        except ValueError:
            pass

    south_lat, west_lon = tile_to_latlon(
        config.TILE_X_MIN, config.TILE_Y_MIN, config.TMS_ZOOM_LEVEL)
    north_lat, _ = tile_to_latlon(
        config.TILE_X_MIN, config.TILE_Y_MAX + 1, config.TMS_ZOOM_LEVEL)
    _, east_lon = tile_to_latlon(
        config.TILE_X_MAX + 1, config.TILE_Y_MIN, config.TMS_ZOOM_LEVEL)
    map_bounds = dict(lat_min=south_lat, lat_max=north_lat,
                      lon_min=west_lon, lon_max=east_lon)
    tee(f"Map bounds: lat [{map_bounds['lat_min']:.4f}, {map_bounds['lat_max']:.4f}], "
        f"lon [{map_bounds['lon_min']:.4f}, {map_bounds['lon_max']:.4f}]")

    raw_df = pd.read_csv(config.IMU_CSV_PATH)
    lat0 = raw_df["latitude"].iloc[0]
    lon0 = raw_df["longitude"].iloc[0]
    alt0 = barometric_altitude(raw_df["barometer_pressure"].iloc[0])
    heading0 = np.degrees(raw_df["heading_magnetic"].iloc[0])
    airspeed0 = (raw_df["airspeed_true"].iloc[0]
                 if "airspeed_true" in raw_df.columns else None)

    live_ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0)
    prev_ts = None
    for i in range(START_ROW + 1):
        row_dict = raw_df.iloc[i].to_dict()
        step_ekf(live_ekf, row_dict, prev_ts)
        prev_ts = row_dict["timestamp"]

    s0 = live_ekf.get_state()
    tee(f"Live EKF initialized at row {START_ROW}: "
        f"({s0['latitude']:.6f}, {s0['longitude']:.6f})  yaw={s0['yaw']:.1f}°")

    aligned = []
    for idx in range(START_ROW, len(imu_log)):
        row = imu_log.iloc[idx]
        ts_rounded = round(row["timestamp"], 3)
        if ts_rounded in frame_map:
            aligned.append((idx, row["timestamp"], frame_map[ts_rounded]))
    tee(f"Aligned {len(aligned)} frame-IMU pairs")

    # ── Cell 3: Init models ──────────────────────────────────────────────────
    tee("\nLoading models...")
    semantic_model = load_semantic_model(config.SEMANTIC_MODEL_PATH, config.DEVICE)
    matcher = initialize_matcher(config.DEVICE, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=config.TMS_ZOOM_LEVEL,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    tee(f"Tiles available: {len(tile_loader.list_tiles())}")

    feature_store = None
    if (hasattr(config, "REFERENCE_FEATURES_PATH")
            and config.REFERENCE_FEATURES_PATH.exists()):
        from Dataset_Preprocessing.feature_store import FeatureStoreLoader
        feature_store = FeatureStoreLoader(
            config.REFERENCE_FEATURES_PATH, device=config.DEVICE)
        feature_store.open()
        tee(f"Feature store: {feature_store.num_tiles} tiles")

    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config,
                                feature_store=feature_store)
    tee("TemporalSearcher initialized")

    # ── Cell 4: Closed-loop (exact copy, with GT) ────────────────────────────
    ekf = copy.deepcopy(live_ekf)
    searcher.frame_count = 0
    searcher.particle_filter = None

    results = []
    ekf_errors_online = []
    ekf_errors_batch  = []
    homo_errors       = []
    trajectory_est    = []
    trajectory_gt     = []
    trajectory_batch  = []
    gate_count        = 0
    prev_ts_ekf = raw_df.iloc[START_ROW]["timestamp"]

    hdr = (f'{"F":>3s} | {"image":<26s} | {"final":>6s} {"batch":>6s} {"homo":>6s} | '
           f'{"CS":>5s} {"inl":>4s} {"SC":>5s} {"gate":>4s} {"R":>5s} | '
           f'{"method":<20s} | {"t":>4s} | note')
    tee(hdr)
    tee("-" * 140)

    t0_total = time.perf_counter()

    for i, (csv_idx, ts, frame_path) in enumerate(aligned[:RUN_N]):
        t0_frame = time.perf_counter()

        row_dict = raw_df.iloc[csv_idx].to_dict()
        step_ekf(ekf, row_dict, prev_ts_ekf)
        prev_ts_ekf = row_dict["timestamp"]

        ekf_state = ekf.get_state()
        ekf_lat, ekf_lon = ekf_state["latitude"], ekf_state["longitude"]
        ekf_yaw = ekf_state["yaw"]

        query_frame = load_image(frame_path)
        vel = np.sqrt(ekf_state["vel_n"] ** 2 + ekf_state["vel_e"] ** 2)
        bank_rad = abs(row_dict.get("bank", 0.0))

        imu_data = {
            "lat": ekf_lat, "lon": ekf_lon,
            "heading": ekf_yaw,
            "pos_sigma": np.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
            "heading_sigma": 15.0,
            "velocity_mps": vel,
            "gyro_z_dps": row_dict.get("gyro_z", 0.0) * (180.0 / np.pi),
            "pitch": row_dict.get("pitch", 0.0),
            "roll": row_dict.get("bank", 0.0),
        }

        result = searcher.process_frame(query_frame, imu_data, timestamp=ts)
        results.append(result)

        gate_pass  = result.get("gate_pass", False)
        homo_pos   = result.get("homo_position")
        vq         = result.get("visual_quality", {})
        cs         = vq.get("CShape", 0)
        ni         = vq.get("inliers", 0)
        sem_conf   = result.get("semantic_confidence") or 0.5
        meta_verified = result.get("meta_tile_verified", False)

        note   = ""
        r_used = None

        if homo_pos is not None and LOOKAHEAD_M > 0:
            h_rad = math.radians(ekf_yaw)
            corr_north = -LOOKAHEAD_M * math.cos(h_rad)
            corr_east  = -LOOKAHEAD_M * math.sin(h_rad)
            homo_pos = (
                homo_pos[0] + corr_north / 111320.0,
                homo_pos[1] + corr_east / (111320.0 * math.cos(math.radians(homo_pos[0]))),
            )

        if gate_pass and homo_pos is not None:
            r_used = R_HIGH if (cs > 0.5 and ni > 100) else R_MED
            if bank_rad > TURN_ROLL_THRESHOLD_RAD:
                r_used *= TURN_R_MULTIPLIER
                note += f"turn({math.degrees(bank_rad):.0f}°) "
            if not meta_verified:
                r_used *= 2.0
                note += "unverified "
            r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
            ekf.update_position(homo_pos[0], homo_pos[1], R_pos_m2=r_used)
            gate_count += 1
        else:
            reasons = []
            if homo_pos is None:
                reasons.append("no_homo")
            if cs <= 0.3:
                reasons.append(f"CS={cs:.2f}<0.3")
            if ni <= 20:
                reasons.append(f"inl={ni}<20")
            note = "SKIP: " + ", ".join(reasons) if reasons else "SKIP"

        corrected = ekf.get_state()
        final_lat, final_lon = corrected["latitude"], corrected["longitude"]

        gt_row  = imu_log.iloc[csv_idx]
        gt_lat  = gt_row["gps_lat"]
        gt_lon  = gt_row["gps_lon"]

        final_err = _hav(final_lat, final_lon, gt_lat, gt_lon)
        batch_err = _hav(gt_row["latitude_est"], gt_row["longitude_est"], gt_lat, gt_lon)
        ekf_errors_online.append(final_err)
        ekf_errors_batch.append(batch_err)
        trajectory_est.append((final_lat, final_lon))
        trajectory_gt.append((gt_lat, gt_lon))
        trajectory_batch.append((gt_row["latitude_est"], gt_row["longitude_est"]))

        homo_err = _hav(homo_pos[0], homo_pos[1], gt_lat, gt_lon) if homo_pos else None
        if homo_err is not None:
            homo_errors.append(homo_err)

        fname  = Path(frame_path).stem
        g_str  = "PASS" if gate_pass else "fail"
        h_str  = f"{homo_err:5.0f}m" if homo_err is not None else "  N/A"
        r_str  = f"{np.sqrt(r_used):4.0f}" if r_used else "   -"
        sc_str = f"{sem_conf:.2f}"
        t_frame = time.perf_counter() - t0_frame
        beat = "✓" if final_err < batch_err else " "

        row_str = (f"F{i:3d} | {fname:<26s} | {final_err:5.0f}m {batch_err:5.0f}m {h_str} | "
                   f"{cs:.3f} {ni:4d} {sc_str:>5s} {g_str:>4s} {r_str} | "
                   f"{result['method']:<20s} | {t_frame:4.1f}s | {note}{beat}")
        tee(row_str)

    elapsed = time.perf_counter() - t0_total

    tee(f'\n{"=" * 140}')
    tee(f"Processed {len(results)} frames in {elapsed:.1f}s "
        f"({elapsed / len(results):.1f}s/frame)")
    tee(f"\n  Online EKF:  mean={np.mean(ekf_errors_online):.1f}m  "
        f"median={np.median(ekf_errors_online):.1f}m  "
        f"min={np.min(ekf_errors_online):.1f}m  "
        f"max={np.max(ekf_errors_online):.1f}m")
    tee(f"  Batch EKF:   mean={np.mean(ekf_errors_batch):.1f}m")
    impr = np.mean(ekf_errors_batch) - np.mean(ekf_errors_online)
    tee(f"  Improvement: {impr:.1f}m ({impr / np.mean(ekf_errors_batch) * 100:.1f}%)")
    tee(f"\n  Gate passes: {gate_count}/{len(results)}")
    if homo_errors:
        tee(f"  Homo-only:   mean={np.mean(homo_errors):.1f}m  "
            f"min={np.min(homo_errors):.1f}m  "
            f"max={np.max(homo_errors):.1f}m  (n={len(homo_errors)})")
    better = sum(1 for e, b in zip(ekf_errors_online, ekf_errors_batch) if e < b)
    tee(f"  Beating batch: {better}/{len(results)} ({better / len(results) * 100:.0f}%)")
    tee(f"  <50m: {sum(1 for e in ekf_errors_online if e < 50)}/{len(results)}  "
        f"<100m: {sum(1 for e in ekf_errors_online if e < 100)}/{len(results)}  "
        f"<150m: {sum(1 for e in ekf_errors_online if e < 150)}/{len(results)}")
    tee(f"  P_pos final: [{ekf.P[8, 8]:.0f}, {ekf.P[9, 9]:.0f}] m²")
    tee(f"  Methods: {dict(Counter(r['method'] for r in results))}")

    # ── Save to file ─────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBaseline saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
