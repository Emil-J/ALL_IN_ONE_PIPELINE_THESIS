"""
runtime/run_pipeline.py  —  Pure Python frame-by-frame runtime pipeline.

No notebook dependency. Reads from a pre-recorded file source (default) or
a live SimConnect source (--source simconnect). Writes results.csv per frame
and run_meta.json at end. No GT comparison, no visualization.

Usage (file mode):
    python runtime/run_pipeline.py --source file --run-id run_001

Usage (live mode):
    python runtime/run_pipeline.py --source simconnect --run-id live_001

Columns in results.csv (21):
    frame_idx, timestamp, image_name, final_lat, final_lon, heading_deg,
    method, gate_pass, search_time_s, cs_shape, inliers, semantic_conf,
    homo_lat, homo_lon, homo_corrected_lat, homo_corrected_lon,
    meta_tile_verified, ekf_pos_sigma, r_used_sqrt, tiles_tested,
    verification_matches
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent           # Pipeline_3_Rev1/
REPO = ROOT.parent                 # All_In_One_Pipeline/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from config import config
from src.tile_utils import TileLoader
from src.image_utils import load_image
from src.geometric_matcher import initialize_matcher
from src.semantic_model import load_semantic_model
from src.temporal_searcher import TemporalSearcher
from src.ekf_ins import (preprocess_imu_csv, ErrorStateEKF,
                          step_ekf, barometric_altitude)

# ── algorithm constants ───────────────────────────────────────────────────────
LOOKAHEAD_M             = 110.0
R_HIGH                  = 30.0 ** 2
R_MED                   = 60.0 ** 2
TURN_ROLL_THRESHOLD_RAD = 0.26
TURN_R_MULTIPLIER       = 3.0

# ── CSV column order ──────────────────────────────────────────────────────────
RESULT_COLUMNS = [
    "frame_idx", "timestamp", "image_name",
    "final_lat", "final_lon", "heading_deg",
    "method", "gate_pass",
    "search_time_s", "cs_shape", "inliers", "semantic_conf",
    "homo_lat", "homo_lon", "homo_corrected_lat", "homo_corrected_lon",
    "meta_tile_verified", "ekf_pos_sigma", "r_used_sqrt",
    "tiles_tested", "verification_matches",
]


def _parse_args():
    p = argparse.ArgumentParser(description="Pipeline 3 runtime")
    p.add_argument("--source", choices=["file", "simconnect"], default="file")
    p.add_argument("--imu-csv",    type=Path, default=None,
                   help="Override IMU CSV path (file mode)")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Override frames directory (file mode)")
    p.add_argument("--run-id",     default=None,
                   help="Run identifier (default: timestamp-based)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Root for output dirs (default: config.RUNS_OUTPUT_DIR)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames")
    p.add_argument("--start-row",  type=int, default=0,
                   help="IMU CSV row to start from (file mode)")
    p.add_argument("--debug",      action="store_true",
                   help="Enable meta-tile PNG saves and history accumulation")
    return p.parse_args()


def _build_run_dir(args) -> Path:
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = "run_" + time.strftime("%Y%m%d_%H%M%S")
    base = args.output_dir or config.RUNS_OUTPUT_DIR
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def _set_deployment_flags(debug: bool):
    config.DEBUG_SAVE_METATILES = debug
    config.ACCUMULATE_HISTORY   = debug


def _init_models():
    semantic_model = load_semantic_model(
        config.SEMANTIC_MODEL_PATH, config.DEVICE)
    matcher = initialize_matcher(config.DEVICE, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=config.TMS_ZOOM_LEVEL,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    feature_store = None
    if (hasattr(config, "REFERENCE_FEATURES_PATH")
            and config.REFERENCE_FEATURES_PATH.exists()):
        from Dataset_Preprocessing.feature_store import FeatureStoreLoader
        feature_store = FeatureStoreLoader(
            config.REFERENCE_FEATURES_PATH, device=config.DEVICE)
        feature_store.open()
    return semantic_model, matcher, tile_loader, feature_store


def _init_ekf(raw_df: pd.DataFrame, start_row: int):
    """Warm up a live EKF through start_row."""
    lat0      = raw_df["latitude"].iloc[0]
    lon0      = raw_df["longitude"].iloc[0]
    alt0      = barometric_altitude(raw_df["barometer_pressure"].iloc[0])
    heading0  = np.degrees(raw_df["heading_magnetic"].iloc[0])
    airspeed0 = (raw_df["airspeed_true"].iloc[0]
                 if "airspeed_true" in raw_df.columns else None)
    ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0)
    prev_ts = None
    for i in range(start_row + 1):
        row_dict = raw_df.iloc[i].to_dict()
        step_ekf(ekf, row_dict, prev_ts)
        prev_ts = row_dict["timestamp"]
    return ekf, prev_ts


def _process_one_frame(frame_idx, csv_idx, ts, frame_path,
                       raw_df, ekf, prev_ts_ekf, searcher) -> tuple:
    """
    Returns (row_dict, final_lat, final_lon, prev_ts_ekf_new, result_row_dict).
    Mutates ekf in place.
    """
    row_dict = raw_df.iloc[csv_idx].to_dict()
    step_ekf(ekf, row_dict, prev_ts_ekf)
    prev_ts_ekf_new = row_dict["timestamp"]

    ekf_state = ekf.get_state()
    ekf_lat   = ekf_state["latitude"]
    ekf_lon   = ekf_state["longitude"]
    ekf_yaw   = ekf_state["yaw"]
    vel       = math.sqrt(ekf_state["vel_n"] ** 2 + ekf_state["vel_e"] ** 2)
    bank_rad  = abs(row_dict.get("bank", 0.0))

    imu_data = {
        "lat":           ekf_lat,
        "lon":           ekf_lon,
        "heading":       ekf_yaw,
        "pos_sigma":     math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
        "heading_sigma": 15.0,
        "velocity_mps":  vel,
        "gyro_z_dps":    row_dict.get("gyro_z", 0.0) * (180.0 / math.pi),
        "pitch":         row_dict.get("pitch", 0.0),
        "roll":          row_dict.get("bank", 0.0),
    }

    query_frame = load_image(frame_path)
    result      = searcher.process_frame(query_frame, imu_data, timestamp=ts)

    gate_pass     = result.get("gate_pass", False)
    homo_pos      = result.get("homo_position")
    vq            = result.get("visual_quality", {})
    cs            = vq.get("CShape", 0.0)
    ni            = vq.get("inliers", 0)
    sem_conf      = result.get("semantic_confidence") or 0.5
    meta_verified = bool(result.get("meta_tile_verified") or False)
    search_time   = result.get("search_time", 0.0)
    tiles_tested  = result.get("tiles_tested", 0)
    ver_matches   = result.get("verification_matches", 0)

    # Lookahead correction
    homo_lat_raw = homo_lon_raw = None
    homo_corr_lat = homo_corr_lon = None
    if homo_pos is not None:
        homo_lat_raw, homo_lon_raw = homo_pos
        if LOOKAHEAD_M > 0:
            h_rad = math.radians(ekf_yaw)
            corr_north = -LOOKAHEAD_M * math.cos(h_rad)
            corr_east  = -LOOKAHEAD_M * math.sin(h_rad)
            homo_corr_lat = (homo_lat_raw
                             + corr_north / 111320.0)
            homo_corr_lon = (homo_lon_raw
                             + corr_east / (111320.0
                                            * math.cos(math.radians(homo_lat_raw))))
            homo_pos = (homo_corr_lat, homo_corr_lon)

    r_used = None
    if gate_pass and homo_pos is not None:
        r_used = R_HIGH if (cs > 0.5 and ni > 100) else R_MED
        if bank_rad > TURN_ROLL_THRESHOLD_RAD:
            r_used *= TURN_R_MULTIPLIER
        if not meta_verified:
            r_used *= 2.0
        r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
        ekf.update_position(homo_pos[0], homo_pos[1], R_pos_m2=r_used)

    final = ekf.get_state()
    final_lat, final_lon = final["latitude"], final["longitude"]
    pos_sigma = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))

    result_row = {
        "frame_idx":          frame_idx,
        "timestamp":          ts,
        "image_name":         Path(frame_path).stem,
        "final_lat":          final_lat,
        "final_lon":          final_lon,
        "heading_deg":        ekf_yaw,
        "method":             result.get("method", ""),
        "gate_pass":          int(gate_pass),
        "search_time_s":      round(search_time, 4),
        "cs_shape":           round(cs, 4),
        "inliers":            ni,
        "semantic_conf":      round(sem_conf, 4),
        "homo_lat":           homo_lat_raw,
        "homo_lon":           homo_lon_raw,
        "homo_corrected_lat": homo_corr_lat,
        "homo_corrected_lon": homo_corr_lon,
        "meta_tile_verified": int(meta_verified),
        "ekf_pos_sigma":      round(pos_sigma, 2),
        "r_used_sqrt":        round(math.sqrt(r_used), 2) if r_used else None,
        "tiles_tested":       tiles_tested,
        "verification_matches": ver_matches,
    }
    return final_lat, final_lon, prev_ts_ekf_new, result_row


def run_file_mode(args, run_dir: Path, run_id: str):
    imu_csv    = args.imu_csv    or config.IMU_CSV_PATH
    frames_dir = args.frames_dir or config.QUERY_FRAMES_DIR
    start_row  = args.start_row

    imu_log  = preprocess_imu_csv(imu_csv)
    raw_df   = pd.read_csv(imu_csv)

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    frame_map   = {}
    for fp in frame_files:
        ts_str = fp.stem.replace("frame_", "")
        try:
            frame_map[round(float(ts_str), 3)] = fp
        except ValueError:
            pass

    aligned = []
    for idx in range(start_row, len(imu_log)):
        row = imu_log.iloc[idx]
        ts_rounded = round(row["timestamp"], 3)
        if ts_rounded in frame_map:
            aligned.append((idx, row["timestamp"], frame_map[ts_rounded]))

    if args.max_frames:
        aligned = aligned[:args.max_frames]

    print(f"[run_pipeline] run_id={run_id}  source=file  frames={len(aligned)}")
    print(f"[run_pipeline] output: {run_dir}")

    ekf, prev_ts_ekf = _init_ekf(raw_df, start_row)
    semantic_model, matcher, tile_loader, feature_store = _init_models()
    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config,
                                feature_store=feature_store)
    searcher.frame_count = 0
    searcher.particle_filter = None

    csv_path = run_dir / "results.csv"
    t0 = time.perf_counter()
    gate_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for i, (csv_idx, ts, frame_path) in enumerate(aligned):
            final_lat, final_lon, prev_ts_ekf, result_row = _process_one_frame(
                i, csv_idx, ts, frame_path,
                raw_df, ekf, prev_ts_ekf, searcher)

            writer.writerow(result_row)
            if result_row["gate_pass"]:
                gate_count += 1

            if (i + 1) % 10 == 0:
                f.flush()

            elapsed = time.perf_counter() - t0
            fps = (i + 1) / elapsed
            print(f"  F{i:4d} | {result_row['image_name']:<26s} | "
                  f"gate={'PASS' if result_row['gate_pass'] else 'fail'} | "
                  f"({final_lat:.6f}, {final_lon:.6f}) | "
                  f"{elapsed:.1f}s ({fps:.2f}fps)")

    total = time.perf_counter() - t0
    _save_meta(run_dir, run_id, "file", len(aligned), gate_count, total,
               str(imu_csv), str(frames_dir))
    print(f"\n[run_pipeline] Done — {len(aligned)} frames in {total:.1f}s "
          f"({total / max(len(aligned), 1):.2f}s/frame)")
    print(f"[run_pipeline] results: {csv_path}")


def run_simconnect_mode(args, run_dir: Path, run_id: str):
    from runtime.simconnect_adapter import SimConnectLiveSource

    print(f"[run_pipeline] run_id={run_id}  source=simconnect")
    print(f"[run_pipeline] output: {run_dir}")

    semantic_model, matcher, tile_loader, feature_store = _init_models()
    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config,
                                feature_store=feature_store)
    searcher.frame_count = 0
    searcher.particle_filter = None

    source = SimConnectLiveSource()
    source.connect()

    # Bootstrap EKF from first good GPS sample
    print("[run_pipeline] Waiting for first valid SimConnect sample...")
    while True:
        row = source.get_latest_row()
        if row and row.get("latitude") and abs(row["latitude"]) > 1.0:
            break
        time.sleep(0.05)

    lat0     = row["latitude"]
    lon0     = row["longitude"]
    alt0     = barometric_altitude(row.get("barometer_pressure", 1013.25))
    heading0 = math.degrees(row.get("heading_magnetic", 0.0))
    ekf      = ErrorStateEKF(lat0, lon0, alt0, heading0, None)
    prev_ts  = row.get("timestamp", time.time())
    print(f"[run_pipeline] EKF bootstrapped: ({lat0:.6f}, {lon0:.6f})  yaw={heading0:.1f}°")

    csv_path   = run_dir / "results.csv"
    max_frames = args.max_frames
    frame_idx  = 0
    gate_count = 0
    t0         = time.perf_counter()
    last_frame_id = None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        try:
            while max_frames is None or frame_idx < max_frames:
                # ── EKF predict at background-thread IMU rate ──────────────
                row = source.get_latest_row()
                if row:
                    step_ekf(ekf, row, prev_ts)
                    prev_ts = row.get("timestamp", time.time())

                # ── Visual processing when a new frame arrives ─────────────
                frame_img, frame_id = source.get_latest_frame()
                if frame_img is None or frame_id == last_frame_id:
                    time.sleep(0.005)
                    continue
                last_frame_id = frame_id

                ekf_state = ekf.get_state()
                ekf_lat   = ekf_state["latitude"]
                ekf_lon   = ekf_state["longitude"]
                ekf_yaw   = ekf_state["yaw"]
                vel       = math.sqrt(ekf_state["vel_n"] ** 2
                                      + ekf_state["vel_e"] ** 2)

                imu_data = {
                    "lat":           ekf_lat,
                    "lon":           ekf_lon,
                    "heading":       ekf_yaw,
                    "pos_sigma":     math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
                    "heading_sigma": 15.0,
                    "velocity_mps":  vel,
                    "gyro_z_dps":    row.get("gyro_z", 0.0) * (180.0 / math.pi)
                                     if row else 0.0,
                    "pitch":         row.get("pitch", 0.0) if row else 0.0,
                    "roll":          row.get("bank", 0.0) if row else 0.0,
                }

                ts = row.get("timestamp", time.time()) if row else time.time()
                result = searcher.process_frame(frame_img, imu_data, timestamp=ts)

                gate_pass     = result.get("gate_pass", False)
                homo_pos      = result.get("homo_position")
                vq            = result.get("visual_quality", {})
                cs            = vq.get("CShape", 0.0)
                ni            = vq.get("inliers", 0)
                sem_conf      = result.get("semantic_confidence") or 0.5
                meta_verified = bool(result.get("meta_tile_verified") or False)

                homo_lat_raw = homo_lon_raw = homo_corr_lat = homo_corr_lon = None
                r_used = None

                if homo_pos is not None:
                    homo_lat_raw, homo_lon_raw = homo_pos
                    h_rad = math.radians(ekf_yaw)
                    corr_north = -LOOKAHEAD_M * math.cos(h_rad)
                    corr_east  = -LOOKAHEAD_M * math.sin(h_rad)
                    homo_corr_lat = (homo_lat_raw
                                     + corr_north / 111320.0)
                    homo_corr_lon = (homo_lon_raw
                                     + corr_east / (111320.0
                                                    * math.cos(math.radians(homo_lat_raw))))
                    homo_pos = (homo_corr_lat, homo_corr_lon)

                    bank_rad = abs(imu_data["roll"])
                    r_used = R_HIGH if (cs > 0.5 and ni > 100) else R_MED
                    if bank_rad > TURN_ROLL_THRESHOLD_RAD:
                        r_used *= TURN_R_MULTIPLIER
                    if not meta_verified:
                        r_used *= 2.0
                    r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
                    if gate_pass:
                        ekf.update_position(homo_pos[0], homo_pos[1],
                                            R_pos_m2=r_used)
                        gate_count += 1

                final = ekf.get_state()
                pos_sigma = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))

                result_row = {
                    "frame_idx":          frame_idx,
                    "timestamp":          ts,
                    "image_name":         f"live_{frame_id}",
                    "final_lat":          final["latitude"],
                    "final_lon":          final["longitude"],
                    "heading_deg":        ekf_yaw,
                    "method":             result.get("method", ""),
                    "gate_pass":          int(gate_pass),
                    "search_time_s":      round(result.get("search_time", 0.0), 4),
                    "cs_shape":           round(cs, 4),
                    "inliers":            ni,
                    "semantic_conf":      round(sem_conf, 4),
                    "homo_lat":           homo_lat_raw,
                    "homo_lon":           homo_lon_raw,
                    "homo_corrected_lat": homo_corr_lat,
                    "homo_corrected_lon": homo_corr_lon,
                    "meta_tile_verified": int(meta_verified),
                    "ekf_pos_sigma":      round(pos_sigma, 2),
                    "r_used_sqrt":        round(math.sqrt(r_used), 2) if r_used else None,
                    "tiles_tested":       result.get("tiles_tested", 0),
                    "verification_matches": result.get("verification_matches", 0),
                }
                writer.writerow(result_row)
                if (frame_idx + 1) % 10 == 0:
                    f.flush()

                elapsed = time.perf_counter() - t0
                print(f"  F{frame_idx:4d} | gate={'PASS' if gate_pass else 'fail'} | "
                      f"({final['latitude']:.6f}, {final['longitude']:.6f}) | "
                      f"{elapsed:.1f}s")
                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[run_pipeline] Interrupted by user")
        finally:
            source.close()

    total = time.perf_counter() - t0
    _save_meta(run_dir, run_id, "simconnect", frame_idx, gate_count, total,
               None, None)
    print(f"\n[run_pipeline] Done — {frame_idx} frames in {total:.1f}s")
    print(f"[run_pipeline] results: {csv_path}")


def _save_meta(run_dir, run_id, source, n_frames, gate_count, elapsed,
               imu_csv, frames_dir):
    meta = {
        "run_id":      run_id,
        "source":      source,
        "n_frames":    n_frames,
        "gate_count":  gate_count,
        "elapsed_s":   round(elapsed, 2),
        "fps":         round(n_frames / max(elapsed, 1e-6), 3),
        "imu_csv":     imu_csv,
        "frames_dir":  frames_dir,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def main():
    args = _parse_args()
    config.ensure_output_dirs()
    _set_deployment_flags(args.debug)

    run_dir, run_id = _build_run_dir(args)

    if args.source == "file":
        run_file_mode(args, run_dir, run_id)
    else:
        run_simconnect_mode(args, run_dir, run_id)


if __name__ == "__main__":
    main()
