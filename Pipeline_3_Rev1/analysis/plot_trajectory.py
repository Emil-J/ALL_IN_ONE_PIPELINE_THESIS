"""
analysis/plot_trajectory.py — Map view + error-over-time (reproduces Cell 6).

Usage:
    python analysis/plot_trajectory.py \
        --run-dir outputs/runs/run_001 \
        --gt-csv  <path-to-imu-gps-log.csv>

Outputs:
    outputs/analysis/<run_id>/trajectory_map.png
    outputs/analysis/<run_id>/error_over_time.png
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config
from src.tile_utils import haversine_distance as _hav
from src.ekf_ins import preprocess_imu_csv


def _parse_args():
    p = argparse.ArgumentParser(description="Plot trajectory and error over time")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--gt-csv",  type=Path, default=None)
    return p.parse_args()


def main():
    args    = _parse_args()
    run_dir = Path(args.run_dir)
    gt_csv  = args.gt_csv or config.IMU_CSV_PATH

    results = pd.read_csv(run_dir / "results.csv")
    imu_log = preprocess_imu_csv(gt_csv)
    imu_log["ts_key"] = imu_log["timestamp"].round(3)
    results["ts_key"] = results["timestamp"].round(3)
    merged = results.merge(
        imu_log[["ts_key", "gps_lat", "gps_lon",
                 "latitude_est", "longitude_est"]],
        on="ts_key", how="left")

    have_gt = merged["gps_lat"].notna()
    m = merged[have_gt]

    final_errs = np.array([
        _hav(r.final_lat, r.final_lon, r.gps_lat, r.gps_lon)
        for _, r in m.iterrows()])
    batch_errs = np.array([
        _hav(r.latitude_est, r.longitude_est, r.gps_lat, r.gps_lon)
        for _, r in m.iterrows()])

    run_id  = run_dir.name
    out_dir = config.ANALYSIS_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Trajectory map ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(m["gps_lon"].values, m["gps_lat"].values,
            "g-", linewidth=1.5, label="GPS ground truth", alpha=0.9)
    ax.plot(m["longitude_est"].values, m["latitude_est"].values,
            "b--", linewidth=1.0, label="Batch EKF", alpha=0.6)
    ax.plot(m["final_lon"].values, m["final_lat"].values,
            "r-", linewidth=1.5, label="Online EKF", alpha=0.9)

    # Gate-pass markers
    gate_rows = m[results["gate_pass"].astype(bool).reindex(m.index, fill_value=False)]
    ax.scatter(gate_rows["final_lon"].values, gate_rows["final_lat"].values,
               c="orange", s=20, zorder=5, label="Visual update")

    # Start/end markers
    ax.plot(m["gps_lon"].iloc[0], m["gps_lat"].iloc[0],
            "g^", markersize=10, label="Start")
    ax.plot(m["gps_lon"].iloc[-1], m["gps_lat"].iloc[-1],
            "gs", markersize=10, label="End")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Trajectory — {run_id}\n"
                 f"Online mean={np.mean(final_errs):.1f}m  "
                 f"Batch mean={np.mean(batch_errs):.1f}m")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "trajectory_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Plot 2: Error over time ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    frame_ids = np.arange(len(final_errs))
    ax.plot(frame_ids, final_errs, "r-",  linewidth=1.2, label="Online EKF", alpha=0.9)
    ax.plot(frame_ids, batch_errs, "b--", linewidth=0.8, label="Batch EKF",  alpha=0.7)
    ax.axhline(50,  color="orange", linestyle=":", linewidth=0.8, label="50m threshold")
    ax.axhline(100, color="gray",   linestyle=":", linewidth=0.8, label="100m threshold")

    # Mark gate-pass frames
    gate_mask = results["gate_pass"].astype(bool).reindex(m.index, fill_value=False).values
    ax.scatter(frame_ids[gate_mask], final_errs[gate_mask],
               c="orange", s=15, zorder=5, alpha=0.7)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Position error (m)")
    ax.set_title(f"Error over time — {run_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out_path = out_dir / "error_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
