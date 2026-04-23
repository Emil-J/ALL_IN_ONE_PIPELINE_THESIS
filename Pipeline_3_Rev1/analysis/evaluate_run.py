"""
analysis/evaluate_run.py — Post-run accuracy analysis.

Loads results.csv from a completed run, joins with GT from the IMU CSV
(via batch EKF), computes per-frame errors, prints a summary matching
the Cell 4 style, and writes summary.txt + summary.json.

Usage:
    python analysis/evaluate_run.py \
        --run-dir outputs/runs/run_001 \
        --gt-csv  <path-to-imu-gps-log.csv>
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from config import config
from src.tile_utils import haversine_distance as _hav
from src.ekf_ins import preprocess_imu_csv


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pipeline run")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Path to run output directory (contains results.csv)")
    p.add_argument("--gt-csv", type=Path, default=None,
                   help="IMU CSV with gps_lat/gps_lon GT columns "
                        "(default: config.IMU_CSV_PATH)")
    return p.parse_args()


def main():
    args   = _parse_args()
    run_dir = Path(args.run_dir)
    gt_csv  = args.gt_csv or config.IMU_CSV_PATH

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")

    results = pd.read_csv(results_path)
    n = len(results)

    # ── Build GT lookup: timestamp → (gt_lat, gt_lon, batch_lat, batch_lon) ──
    imu_log = preprocess_imu_csv(gt_csv)   # adds latitude_est / longitude_est

    # Round timestamps to 3dp for join
    imu_log["ts_key"]  = imu_log["timestamp"].round(3)
    results["ts_key"]  = results["timestamp"].round(3)

    merged = results.merge(
        imu_log[["ts_key", "gps_lat", "gps_lon",
                 "latitude_est", "longitude_est"]],
        on="ts_key", how="left")

    have_gt = merged["gps_lat"].notna()

    final_errs = []
    batch_errs = []
    homo_errs  = []

    for _, row in merged[have_gt].iterrows():
        gt_lat, gt_lon = row["gps_lat"], row["gps_lon"]
        final_errs.append(_hav(row["final_lat"], row["final_lon"], gt_lat, gt_lon))
        batch_errs.append(_hav(row["latitude_est"], row["longitude_est"], gt_lat, gt_lon))
        if pd.notna(row.get("homo_corrected_lat")) and pd.notna(row.get("homo_corrected_lon")):
            homo_errs.append(_hav(row["homo_corrected_lat"], row["homo_corrected_lon"],
                                  gt_lat, gt_lon))

    final_errs = np.array(final_errs)
    batch_errs = np.array(batch_errs)
    homo_errs  = np.array(homo_errs) if homo_errs else np.array([])

    gate_count = int(results["gate_pass"].sum())

    lines = []

    def tee(msg=""):
        print(msg)
        lines.append(msg)

    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        tee(f"Run: {meta.get('run_id', run_dir.name)}")
        tee(f"  source={meta.get('source')}  frames={meta.get('n_frames')}  "
            f"elapsed={meta.get('elapsed_s')}s  fps={meta.get('fps')}")
    else:
        tee(f"Run dir: {run_dir}")

    tee(f"\nFrames with GT: {len(final_errs)}/{n}")
    tee(f"\nOnline EKF errors:")
    tee(f"  mean   = {np.mean(final_errs):.1f} m")
    tee(f"  median = {np.median(final_errs):.1f} m")
    tee(f"  min    = {np.min(final_errs):.1f} m")
    tee(f"  max    = {np.max(final_errs):.1f} m")
    tee(f"  std    = {np.std(final_errs):.1f} m")

    if len(batch_errs):
        tee(f"\nBatch EKF errors:")
        tee(f"  mean   = {np.mean(batch_errs):.1f} m")
        impr = np.mean(batch_errs) - np.mean(final_errs)
        tee(f"  Improvement over batch: {impr:.1f} m "
            f"({impr / np.mean(batch_errs) * 100:.1f}%)")

    if len(homo_errs):
        tee(f"\nHomography-only errors (n={len(homo_errs)}):")
        tee(f"  mean={np.mean(homo_errs):.1f} m  min={np.min(homo_errs):.1f} m  "
            f"max={np.max(homo_errs):.1f} m")

    tee(f"\nGate passes: {gate_count}/{n}")
    better = int(np.sum(final_errs < batch_errs[:len(final_errs)]))
    tee(f"Beating batch: {better}/{len(final_errs)}")

    for thresh in config.EVALUATION_THRESHOLDS:
        cnt = int(np.sum(final_errs <= thresh))
        tee(f"  <{thresh:4d}m : {cnt}/{len(final_errs)} "
            f"({cnt / max(len(final_errs), 1) * 100:.0f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    run_id = run_dir.name
    out_dir = config.ANALYSIS_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_txt = out_dir / "summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    summary_json = out_dir / "summary.json"
    summary_data = {
        "run_id":          run_id,
        "n_frames":        n,
        "n_with_gt":       len(final_errs),
        "gate_passes":     gate_count,
        "online_mean_m":   float(np.mean(final_errs)),
        "online_median_m": float(np.median(final_errs)),
        "online_min_m":    float(np.min(final_errs)),
        "online_max_m":    float(np.max(final_errs)),
        "online_std_m":    float(np.std(final_errs)),
        "batch_mean_m":    float(np.mean(batch_errs)) if len(batch_errs) else None,
        "homo_mean_m":     float(np.mean(homo_errs))  if len(homo_errs)  else None,
        "better_than_batch": better,
        "pct_under_50m":   float(np.mean(final_errs <= 50) * 100),
        "pct_under_100m":  float(np.mean(final_errs <= 100) * 100),
    }
    summary_json.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Also save per-frame error table
    merged["final_err_m"] = pd.Series(
        [_hav(r.final_lat, r.final_lon, r.gps_lat, r.gps_lon)
         if pd.notna(r.gps_lat) else None
         for _, r in merged.iterrows()])
    merged.to_csv(out_dir / "frames.csv", index=False)

    tee(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
