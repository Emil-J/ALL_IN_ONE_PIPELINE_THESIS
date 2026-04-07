"""
Phase B1 Diagnostic Script — Rotation + Dual Homography + Visual Measurements.

For the same 10 Phase A test frames, produces:
  1. Unrotated vs heading-rotated matching (baseline comparison).
  2. Dual homography evaluation (MAGSAC vs DLT) on both rotated and unrotated.
  3. Five visual measurement extraction methods per frame.
  4. Per-method ground-truth error comparison.
  5. Winner selection analysis.

Outputs:
  outputs/phase_b1/summary.csv           — one row per frame
  outputs/phase_b1/measurements.csv      — one row per (frame, method)
  outputs/phase_b1/frame_NNN.json        — full detail per frame
"""

import sys, os, json, time, math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.tile_utils import (
    TileLoader, find_tiles_within_radius, tile_to_latlon,
    latlon_to_tile_float, haversine_distance,
)
from src.image_utils import load_image
from src.geometric_matcher import initialize_matcher
from src.ekf_ins import preprocess_imu_csv
from src.visual_measurement import (
    rotate_image, get_rotation_inverse,
    compute_shape_confidence, compute_dual_homography,
    extract_visual_measurements,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

NUM_TEST_FRAMES = 10
SEARCH_RADIUS_M = 500.0
RANSAC_THRESH = config.RANSAC_REPROJ_THRESH  # 8.0
OUTPUT_DIR = config.OUTPUT_DIR / "phase_b1"
TILE_PX = config.TMS_TILE_SIZE_PX
ZOOM = config.TMS_ZOOM_LEVEL


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def build_meta_tile(top3: List[Dict], tile_loader: TileLoader
                    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int]], Dict]:
    """Build meta-tile canvas from top-3 tiles."""
    if not top3:
        return None, [], {}

    xs = [t["tx"] for t in top3]
    ys = [t["ty"] for t in top3]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1

    canvas = np.zeros((rows * TILE_PX, cols * TILE_PX, 3), dtype=np.uint8)
    filled = 0
    for t in top3:
        img = tile_loader.load_aerial(t["tx"], t["ty"])
        if img is None:
            continue
        col = t["tx"] - x_min
        row = t["ty"] - y_min
        canvas[row * TILE_PX:(row + 1) * TILE_PX,
               col * TILE_PX:(col + 1) * TILE_PX] = img
        filled += 1

    top3_tiles = [(t["tx"], t["ty"], t["num_matches"]) for t in top3]
    occupancy = {
        "grid_rows": rows, "grid_cols": cols,
        "filled_cells": filled, "total_cells": rows * cols,
        "fill_ratio": filled / (rows * cols),
    }
    return canvas, top3_tiles, occupancy


def match_all_tiles(query: np.ndarray, candidates: List[Tuple[int, int]],
                    matcher, tile_loader: TileLoader) -> List[Dict]:
    """Match query against all candidate tiles. Return sorted by num_matches desc."""
    results = []
    for tx, ty in candidates:
        tile_img = tile_loader.load_aerial(tx, ty)
        if tile_img is None:
            continue
        res = matcher.match(query, tile_img)
        results.append({
            "tx": tx, "ty": ty,
            "num_matches": res["num_matches"],
            "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
        })
    results.sort(key=lambda r: r["num_matches"], reverse=True)
    return results


def serialise(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE B1 DIAGNOSTICS — Rotation + Dual Homography + Measurements")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading IMU data and building frame list...")
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

    # Auto-detect START_ROW
    south_lat, west_lon = tile_to_latlon(config.TILE_X_MIN, config.TILE_Y_MIN, ZOOM)
    north_lat, _ = tile_to_latlon(config.TILE_X_MIN, config.TILE_Y_MAX + 1, ZOOM)
    _, east_lon = tile_to_latlon(config.TILE_X_MAX + 1, config.TILE_Y_MIN, ZOOM)

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
    print(f"  START_ROW={START_ROW}, aligned {len(aligned)} frames")

    # Same 10 frames as Phase A
    indices = np.linspace(0, len(aligned) - 1, NUM_TEST_FRAMES, dtype=int)
    test_frames = [aligned[i] for i in indices]
    print(f"  Selected {len(test_frames)} test frames at indices: {indices.tolist()}")

    # ── Initialize ──
    print("\n[2/5] Initializing matcher and tile loader...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    matcher = initialize_matcher(device, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=ZOOM,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    print(f"  Matcher ready, {len(tile_loader.list_tiles())} tiles available")

    # ── Per-frame diagnostics ──
    print(f"\n[3/5] Running per-frame diagnostics on {len(test_frames)} frames...")
    summary_rows = []
    measurement_rows = []

    for fi, (csv_idx, ts, frame_path) in enumerate(test_frames):
        print(f"\n{'='*60}")
        print(f"  Frame {fi}/{len(test_frames)-1}  ts={ts:.3f}  {frame_path.name}")
        print(f"{'='*60}")
        row = imu_log.iloc[csv_idx]
        query_frame = load_image(frame_path)
        qh, qw = query_frame.shape[:2]

        ekf_lat = row['latitude_est']
        ekf_lon = row['longitude_est']
        heading_deg = row['yaw_deg']
        gt_lat = row['gps_lat']
        gt_lon = row['gps_lon']
        ekf_error = haversine_distance(ekf_lat, ekf_lon, gt_lat, gt_lon)

        # IMU attitude (approximate from logged values)
        pitch_rad = row.get('pitch', 0.0) if 'pitch' in row.index else 0.0
        roll_rad = row.get('bank', 0.0) if 'bank' in row.index else 0.0

        print(f"  EKF=({ekf_lat:.5f},{ekf_lon:.5f})  GT=({gt_lat:.5f},{gt_lon:.5f})  "
              f"EKF_err={ekf_error:.1f}m  heading={heading_deg:.1f}°")

        # ── Find candidates ──
        candidates = find_tiles_within_radius(
            ekf_lat, ekf_lon, SEARCH_RADIUS_M,
            zoom=ZOOM,
            x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
            y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
        )
        print(f"  Candidates: {len(candidates)} tiles")

        # ── UNROTATED: match, build meta-tile, dual homography ──
        print("  --- Unrotated pipeline ---")
        tile_matches_unrot = match_all_tiles(query_frame, candidates, matcher, tile_loader)
        top1_unrot = tile_matches_unrot[0] if tile_matches_unrot else None
        print(f"  Top-1: {top1_unrot['num_matches'] if top1_unrot else 0} matches")

        meta_unrot, top3_tiles_unrot, occ_unrot = build_meta_tile(
            tile_matches_unrot[:3], tile_loader)

        dual_unrot = None
        measurements_unrot = {}
        if meta_unrot is not None:
            meta_match_unrot = matcher.match(query_frame, meta_unrot)
            print(f"  Meta-tile match: {meta_match_unrot['num_matches']} matches")

            if meta_match_unrot["num_matches"] >= 4:
                src_pts = meta_match_unrot["keypoints1"][meta_match_unrot["matches"][:, 0]]
                dst_pts = meta_match_unrot["keypoints2"][meta_match_unrot["matches"][:, 1]]

                dual_unrot = compute_dual_homography(
                    src_pts, dst_pts, qw, qh, RANSAC_THRESH)

                if dual_unrot["winner"]:
                    print(f"  Dual winner: {dual_unrot['winner']}")
                    for branch_name in ["dlt", "magsac"]:
                        br = dual_unrot[branch_name]
                        if br:
                            print(f"    {branch_name:8s}: inliers={br['inliers']:3d}  "
                                  f"CShape={br['CShape']:.3f}  convex={br['convex']}")

                    measurements_unrot = extract_visual_measurements(
                        dual_unrot["winner_H"], dual_unrot["winner_mask"],
                        src_pts, dst_pts, qw, qh,
                        top3_tiles_unrot,
                        tile_px=TILE_PX, zoom=ZOOM,
                        pitch_rad=pitch_rad, roll_rad=roll_rad,
                    )

                    # Evaluate vs ground truth
                    print(f"  Measurements (unrotated):")
                    for mname, mdata in measurements_unrot.items():
                        if mdata["valid"] and mdata["latlon"][0] is not None:
                            err = haversine_distance(
                                mdata["latlon"][0], mdata["latlon"][1],
                                gt_lat, gt_lon)
                            mdata["error_m"] = err
                            print(f"    {mname:25s}: ({mdata['latlon'][0]:.5f},{mdata['latlon'][1]:.5f})  "
                                  f"err={err:.1f}m  valid={mdata['valid']}")
                        else:
                            mdata["error_m"] = None
                            print(f"    {mname:25s}: INVALID")

        # ── ROTATED: rotate query by heading, match, build meta-tile, dual homography ──
        print("  --- Rotated pipeline (heading rotation) ---")
        rotation_angle = -heading_deg
        query_rotated, M_fwd = rotate_image(query_frame, rotation_angle)
        rh, rw = query_rotated.shape[:2]
        print(f"  Rotation: {rotation_angle:.1f}° → {rw}x{rh}")

        tile_matches_rot = match_all_tiles(query_rotated, candidates, matcher, tile_loader)
        top1_rot = tile_matches_rot[0] if tile_matches_rot else None
        print(f"  Top-1: {top1_rot['num_matches'] if top1_rot else 0} matches")

        meta_rot, top3_tiles_rot, occ_rot = build_meta_tile(
            tile_matches_rot[:3], tile_loader)

        dual_rot = None
        measurements_rot = {}
        if meta_rot is not None:
            meta_match_rot = matcher.match(query_rotated, meta_rot)
            print(f"  Meta-tile match: {meta_match_rot['num_matches']} matches")

            if meta_match_rot["num_matches"] >= 4:
                src_pts_r = meta_match_rot["keypoints1"][meta_match_rot["matches"][:, 0]]
                dst_pts_r = meta_match_rot["keypoints2"][meta_match_rot["matches"][:, 1]]

                dual_rot = compute_dual_homography(
                    src_pts_r, dst_pts_r, rw, rh, RANSAC_THRESH)

                if dual_rot["winner"]:
                    print(f"  Dual winner: {dual_rot['winner']}")
                    for branch_name in ["dlt", "magsac"]:
                        br = dual_rot[branch_name]
                        if br:
                            print(f"    {branch_name:8s}: inliers={br['inliers']:3d}  "
                                  f"CShape={br['CShape']:.3f}  convex={br['convex']}")

                    measurements_rot = extract_visual_measurements(
                        dual_rot["winner_H"], dual_rot["winner_mask"],
                        src_pts_r, dst_pts_r, rw, rh,
                        top3_tiles_rot,
                        tile_px=TILE_PX, zoom=ZOOM,
                        pitch_rad=pitch_rad, roll_rad=roll_rad,
                    )

                    print(f"  Measurements (rotated):")
                    for mname, mdata in measurements_rot.items():
                        if mdata["valid"] and mdata["latlon"][0] is not None:
                            err = haversine_distance(
                                mdata["latlon"][0], mdata["latlon"][1],
                                gt_lat, gt_lon)
                            mdata["error_m"] = err
                            print(f"    {mname:25s}: ({mdata['latlon'][0]:.5f},{mdata['latlon'][1]:.5f})  "
                                  f"err={err:.1f}m  valid={mdata['valid']}")
                        else:
                            mdata["error_m"] = None
                            print(f"    {mname:25s}: INVALID")

        # ── Compile frame summary ──
        frame_summary = {
            "frame_idx": fi,
            "csv_idx": int(csv_idx),
            "timestamp": ts,
            "frame_name": frame_path.name,
            "heading_deg": heading_deg,
            "ekf_lat": ekf_lat, "ekf_lon": ekf_lon,
            "gt_lat": gt_lat, "gt_lon": gt_lon,
            "ekf_error_m": ekf_error,
            "n_candidates": len(candidates),
            # Unrotated
            "unrot_top1_matches": top1_unrot["num_matches"] if top1_unrot else 0,
            "unrot_meta_matches": meta_match_unrot["num_matches"] if meta_unrot is not None else 0,
            "unrot_winner": dual_unrot["winner"] if dual_unrot else None,
            # Rotated
            "rot_top1_matches": top1_rot["num_matches"] if top1_rot else 0,
            "rot_meta_matches": meta_match_rot["num_matches"] if meta_rot is not None else 0,
            "rot_winner": dual_rot["winner"] if dual_rot else None,
        }

        # Add per-branch homography stats
        for prefix, dual in [("unrot", dual_unrot), ("rot", dual_rot)]:
            for branch in ["dlt", "magsac"]:
                br = dual[branch] if dual else None
                if br:
                    frame_summary[f"{prefix}_{branch}_inliers"] = br["inliers"]
                    frame_summary[f"{prefix}_{branch}_reproj"] = br["reproj_median"]
                    frame_summary[f"{prefix}_{branch}_cshape"] = br["CShape"]
                    frame_summary[f"{prefix}_{branch}_convex"] = br["convex"]
                else:
                    frame_summary[f"{prefix}_{branch}_inliers"] = None
                    frame_summary[f"{prefix}_{branch}_reproj"] = None
                    frame_summary[f"{prefix}_{branch}_cshape"] = None
                    frame_summary[f"{prefix}_{branch}_convex"] = None

        # Best measurement per variant
        for prefix, meas_dict in [("unrot", measurements_unrot), ("rot", measurements_rot)]:
            best_err = float('inf')
            best_method = None
            for mname, mdata in meas_dict.items():
                err = mdata.get("error_m")
                if err is not None and err < best_err:
                    best_err = err
                    best_method = mname
                # Record per-method error
                frame_summary[f"{prefix}_{mname}_error_m"] = err
            frame_summary[f"{prefix}_best_method"] = best_method
            frame_summary[f"{prefix}_best_error_m"] = best_err if best_err < float('inf') else None

        summary_rows.append(frame_summary)

        # Measurement detail rows
        for prefix, meas_dict in [("unrot", measurements_unrot), ("rot", measurements_rot)]:
            for mname, mdata in meas_dict.items():
                measurement_rows.append({
                    "frame_idx": fi,
                    "variant": prefix,
                    "method": mname,
                    "valid": mdata["valid"],
                    "px_x": mdata["px"][0] if mdata["px"][0] is not None else None,
                    "px_y": mdata["px"][1] if mdata["px"][1] is not None else None,
                    "lat": mdata["latlon"][0],
                    "lon": mdata["latlon"][1],
                    "error_m": mdata.get("error_m"),
                    "ekf_error_m": ekf_error,
                })

        # Save per-frame JSON (make it JSON-serialisable)
        frame_detail = {
            "summary": frame_summary,
            "unrotated": {
                "top3_tiles": [(t["tx"], t["ty"], t["num_matches"]) for t in tile_matches_unrot[:3]],
                "occupancy": occ_unrot,
                "dual_winner": dual_unrot["winner"] if dual_unrot else None,
                "dlt": {k: v for k, v in (dual_unrot["dlt"] or {}).items()
                        if k not in ("H", "inlier_mask", "reproj_all")} if dual_unrot and dual_unrot["dlt"] else None,
                "magsac": {k: v for k, v in (dual_unrot["magsac"] or {}).items()
                           if k not in ("H", "inlier_mask", "reproj_all")} if dual_unrot and dual_unrot["magsac"] else None,
                "measurements": {k: {kk: vv for kk, vv in v.items()
                                      if kk not in ("nadir_shift_px",)}
                                 for k, v in measurements_unrot.items()},
            },
            "rotated": {
                "rotation_angle": rotation_angle,
                "rotated_size": [rw, rh],
                "top3_tiles": [(t["tx"], t["ty"], t["num_matches"]) for t in tile_matches_rot[:3]],
                "occupancy": occ_rot,
                "dual_winner": dual_rot["winner"] if dual_rot else None,
                "dlt": {k: v for k, v in (dual_rot["dlt"] or {}).items()
                        if k not in ("H", "inlier_mask", "reproj_all")} if dual_rot and dual_rot["dlt"] else None,
                "magsac": {k: v for k, v in (dual_rot["magsac"] or {}).items()
                           if k not in ("H", "inlier_mask", "reproj_all")} if dual_rot and dual_rot["magsac"] else None,
                "measurements": {k: {kk: vv for kk, vv in v.items()
                                      if kk not in ("nadir_shift_px",)}
                                 for k, v in measurements_rot.items()},
            },
        }
        with open(OUTPUT_DIR / f"frame_{fi:03d}.json", "w") as f:
            json.dump(frame_detail, f, indent=2, default=serialise)

    # ── Save CSVs ──
    print(f"\n[4/5] Saving results...")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"  summary.csv: {len(summary_df)} rows")

    meas_df = pd.DataFrame(measurement_rows)
    meas_df.to_csv(OUTPUT_DIR / "measurements.csv", index=False)
    print(f"  measurements.csv: {len(meas_df)} rows")

    # ── Summary statistics ──
    print(f"\n[5/5] Summary Statistics")
    print("=" * 70)

    print(f"\n  EKF baseline: mean={summary_df['ekf_error_m'].mean():.1f}m, "
          f"median={summary_df['ekf_error_m'].median():.1f}m")

    print(f"\n  === Matching (top-1 tile) ===")
    print(f"  Unrotated: mean={summary_df['unrot_top1_matches'].mean():.1f} matches")
    print(f"  Rotated:   mean={summary_df['rot_top1_matches'].mean():.1f} matches")
    delta = summary_df['rot_top1_matches'] - summary_df['unrot_top1_matches']
    print(f"  Delta:     mean={delta.mean():+.1f}")

    print(f"\n  === Dual Homography ===")
    for prefix in ["unrot", "rot"]:
        print(f"  [{prefix}]")
        for branch in ["dlt", "magsac"]:
            col = f"{prefix}_{branch}_inliers"
            valid = summary_df[col].dropna()
            if len(valid) > 0:
                cs_col = f"{prefix}_{branch}_cshape"
                cvx_col = f"{prefix}_{branch}_convex"
                print(f"    {branch:8s}: inliers mean={valid.mean():.1f}  "
                      f"CShape={summary_df[cs_col].dropna().mean():.3f}  "
                      f"convex={summary_df[cvx_col].dropna().sum():.0f}/{len(valid)}")
        winners = summary_df[f"{prefix}_winner"].value_counts()
        print(f"    Winners: {dict(winners)}")

    print(f"\n  === Measurement Methods (errors in meters) ===")
    methods = ["projected_center", "inlier_centroid", "trimmed_centroid",
               "nadir_corrected", "weighted_centroid"]
    for prefix in ["unrot", "rot"]:
        print(f"  [{prefix}]")
        for m in methods:
            col = f"{prefix}_{m}_error_m"
            if col in summary_df.columns:
                valid = summary_df[col].dropna()
                if len(valid) > 0:
                    print(f"    {m:25s}: n={len(valid)}  mean={valid.mean():.1f}m  "
                          f"median={valid.median():.1f}m  min={valid.min():.1f}m")

    # Best method comparison
    print(f"\n  === Best Method Per Frame ===")
    for prefix in ["unrot", "rot"]:
        col = f"{prefix}_best_method"
        err_col = f"{prefix}_best_error_m"
        valid = summary_df[summary_df[err_col].notna()]
        if len(valid) > 0:
            print(f"  [{prefix}] best error: mean={valid[err_col].mean():.1f}m  "
                  f"median={valid[err_col].median():.1f}m")
            print(f"    Methods chosen: {dict(valid[col].value_counts())}")

    # vs EKF comparison
    print(f"\n  === Improvement over EKF ===")
    for prefix in ["unrot", "rot"]:
        err_col = f"{prefix}_best_error_m"
        valid = summary_df.dropna(subset=[err_col])
        if len(valid) > 0:
            better = (valid[err_col] < valid["ekf_error_m"]).sum()
            ekf_mean = valid["ekf_error_m"].mean()
            best_mean = valid[err_col].mean()
            print(f"  [{prefix}] {better}/{len(valid)} frames beat EKF  "
                  f"(visual={best_mean:.1f}m vs EKF={ekf_mean:.1f}m, "
                  f"delta={best_mean - ekf_mean:+.1f}m)")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
