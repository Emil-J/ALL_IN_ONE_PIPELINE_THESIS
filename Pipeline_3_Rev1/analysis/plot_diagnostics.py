"""
analysis/plot_diagnostics.py — 3-panel diagnostics (reproduces Cell 7).

Plots per-frame: CShape score, inlier count, and homography error.

Usage:
    python analysis/plot_diagnostics.py \
        --run-dir outputs/runs/run_001 \
        --gt-csv  <path-to-imu-gps-log.csv>

Outputs:
    outputs/analysis/<run_id>/diagnostics.png
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
    p = argparse.ArgumentParser(description="Plot pipeline diagnostics")
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
        imu_log[["ts_key", "gps_lat", "gps_lon"]],
        on="ts_key", how="left")

    have_gt  = merged["gps_lat"].notna()
    m        = merged[have_gt].reset_index(drop=True)
    frame_ids = np.arange(len(m))

    # Homography error (corrected if available, else raw)
    homo_err = []
    for _, row in m.iterrows():
        lat = row.get("homo_corrected_lat") or row.get("homo_lat")
        lon = row.get("homo_corrected_lon") or row.get("homo_lon")
        if pd.notna(lat) and pd.notna(lon):
            homo_err.append(_hav(lat, lon, row["gps_lat"], row["gps_lon"]))
        else:
            homo_err.append(np.nan)
    homo_err = np.array(homo_err)

    run_id  = run_dir.name
    out_dir = config.ANALYSIS_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_mask = m["gate_pass"].astype(bool).values

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Diagnostics — {run_id}", fontsize=12)

    # Panel 1: CShape
    ax = axes[0]
    ax.plot(frame_ids, m["cs_shape"].values, "b-", linewidth=1.0, alpha=0.8)
    ax.axhline(config.QUALITY_GATE_CSHAPE, color="r", linestyle="--",
               linewidth=0.8, label=f"gate={config.QUALITY_GATE_CSHAPE}")
    ax.scatter(frame_ids[gate_mask], m["cs_shape"].values[gate_mask],
               c="orange", s=15, zorder=5, alpha=0.7, label="gate pass")
    ax.set_ylabel("CShape")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Inliers
    ax = axes[1]
    ax.bar(frame_ids, m["inliers"].values, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(config.QUALITY_GATE_INLIERS, color="r", linestyle="--",
               linewidth=0.8, label=f"gate={config.QUALITY_GATE_INLIERS}")
    ax.set_ylabel("Inliers")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Homography error
    ax = axes[2]
    valid = ~np.isnan(homo_err)
    ax.scatter(frame_ids[valid], homo_err[valid],
               c=np.where(gate_mask[valid], "orange", "gray"),
               s=20, alpha=0.8, zorder=5)
    ax.axhline(50,  color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(100, color="gray",   linestyle=":", linewidth=0.8)
    ax.set_ylabel("Homo error (m)")
    ax.set_xlabel("Frame index")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "diagnostics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
