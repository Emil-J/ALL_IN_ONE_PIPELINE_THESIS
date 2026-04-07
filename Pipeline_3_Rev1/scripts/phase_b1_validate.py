"""
Phase B1 Pipeline Validation.

Runs the full updated pipeline (with heading rotation + dual homography +
visual measurement extraction) on the same 10 test frames and compares
against EKF baseline and Phase A results.
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.tile_utils import (
    TileLoader, tile_to_latlon, haversine_distance,
)
from src.image_utils import load_image
from src.geometric_matcher import initialize_matcher
from src.ekf_ins import preprocess_imu_csv
from src.semantic_model import SemanticModel
from src.temporal_searcher import TemporalSearcher

OUTPUT_DIR = config.OUTPUT_DIR / "phase_b1"
NUM_TEST_FRAMES = 10


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE B1 PIPELINE VALIDATION — Full Pipeline on 10 Test Frames")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    imu_log = preprocess_imu_csv(config.IMU_CSV_PATH)

    frame_dir = config.QUERY_FRAMES_DIR
    frame_files = sorted(frame_dir.glob('frame_*.jpg'))
    frame_map = {}
    for fp in frame_files:
        ts_str = fp.stem.replace('frame_', '')
        try:
            frame_map[round(float(ts_str), 3)] = fp
        except ValueError:
            continue

    south_lat, west_lon = tile_to_latlon(config.TILE_X_MIN, config.TILE_Y_MIN, config.TMS_ZOOM_LEVEL)
    north_lat, _ = tile_to_latlon(config.TILE_X_MIN, config.TILE_Y_MAX + 1, config.TMS_ZOOM_LEVEL)
    _, east_lon = tile_to_latlon(config.TILE_X_MAX + 1, config.TILE_Y_MIN, config.TMS_ZOOM_LEVEL)

    START_ROW = 0
    for idx in range(len(imu_log)):
        row = imu_log.iloc[idx]
        if (south_lat <= row['gps_lat'] <= north_lat and
                west_lon <= row['gps_lon'] <= east_lon):
            START_ROW = idx
            break

    aligned = []
    for idx in range(START_ROW, len(imu_log)):
        row = imu_log.iloc[idx]
        ts_rounded = round(row['timestamp'], 3)
        if ts_rounded in frame_map:
            aligned.append((idx, row['timestamp'], frame_map[ts_rounded]))
        if len(aligned) >= 300:
            break

    indices = np.linspace(0, len(aligned) - 1, NUM_TEST_FRAMES, dtype=int)
    test_frames = [aligned[i] for i in indices]
    print(f"  {len(test_frames)} test frames from {len(aligned)} aligned")

    # Initialize pipeline
    print("\n[2/4] Initializing pipeline...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    matcher = initialize_matcher(device, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=config.TMS_ZOOM_LEVEL,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    semantic_model = SemanticModel(config.SEMANTIC_MODEL_PATH, device=device)

    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config)

    # Process frames
    print(f"\n[3/4] Processing {len(test_frames)} frames...")
    results = []
    for fi, (csv_idx, ts, frame_path) in enumerate(test_frames):
        row = imu_log.iloc[csv_idx]
        query_frame = load_image(frame_path)

        ekf_lat = row['latitude_est']
        ekf_lon = row['longitude_est']
        ekf_yaw_deg = row['yaw_deg']
        gt_lat = row['gps_lat']
        gt_lon = row['gps_lon']

        if 'vel_n' in row.index and 'vel_e' in row.index:
            vel = np.sqrt(row['vel_n'] ** 2 + row['vel_e'] ** 2)
        else:
            vel = 20.0

        gyro_z_dps = row.get('gyro_z', 0.0) * (180.0 / np.pi)

        imu_data = {
            'lat': ekf_lat,
            'lon': ekf_lon,
            'heading': ekf_yaw_deg,
            'pos_sigma': 100.0,
            'heading_sigma': 15.0,
            'velocity_mps': vel,
            'gyro_z_dps': gyro_z_dps,
            'pitch': row.get('pitch', 0.0),
            'roll': row.get('bank', 0.0),
        }

        t0 = time.perf_counter()
        result = searcher.process_frame(query_frame, imu_data, timestamp=ts)
        elapsed = time.perf_counter() - t0

        pos = result.get("position", (None, None))
        if pos[0] is not None:
            pipeline_error = haversine_distance(pos[0], pos[1], gt_lat, gt_lon)
        else:
            pipeline_error = None

        ekf_error = haversine_distance(ekf_lat, ekf_lon, gt_lat, gt_lon)

        results.append({
            "frame_idx": fi,
            "csv_idx": int(csv_idx),
            "timestamp": ts,
            "method": result.get("method"),
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "ekf_lat": ekf_lat, "ekf_lon": ekf_lon,
            "est_lat": pos[0], "est_lon": pos[1],
            "ekf_error_m": ekf_error,
            "pipeline_error_m": pipeline_error,
            "score": result.get("score", 0),
            "tiles_tested": result.get("tiles_tested", 0),
            "elapsed_s": elapsed,
        })

        delta = (pipeline_error - ekf_error) if pipeline_error else None
        delta_str = f"{delta:+.1f}m" if delta is not None else "N/A"
        print(f"  Frame {fi}: EKF={ekf_error:.1f}m  Pipeline={pipeline_error:.1f}m  "
              f"Δ={delta_str}  method={result['method']}  score={result.get('score', 0)}  "
              f"t={elapsed:.1f}s")

    # Save and summarize
    print(f"\n[4/4] Results Summary")
    print("=" * 70)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "pipeline_validation.csv", index=False)

    valid = df.dropna(subset=["pipeline_error_m"])
    ekf_mean = valid["ekf_error_m"].mean()
    pipe_mean = valid["pipeline_error_m"].mean()
    beats = (valid["pipeline_error_m"] < valid["ekf_error_m"]).sum()

    print(f"  EKF baseline:    mean={ekf_mean:.1f}m  median={valid['ekf_error_m'].median():.1f}m")
    print(f"  Pipeline (B1):   mean={pipe_mean:.1f}m  median={valid['pipeline_error_m'].median():.1f}m")
    print(f"  Improvement:     {ekf_mean - pipe_mean:+.1f}m ({(ekf_mean - pipe_mean)/ekf_mean*100:.1f}%)")
    print(f"  Beats EKF:       {beats}/{len(valid)}")
    print(f"  Per-frame errors: {[f'{e:.1f}' for e in valid['pipeline_error_m']]}")
    print(f"\n  Saved to: {OUTPUT_DIR / 'pipeline_validation.csv'}")

    searcher.close()


if __name__ == "__main__":
    main()
