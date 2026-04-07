"""
Phase A Diagnostic Script — Instrumentation & Orientation Experiments.

For N test frames, produces:
  1. Per-tile match statistics (candidate tiles, chosen tiles, scores).
  2. Meta-tile occupancy (how many grid cells are filled vs black).
  3. Projected query quad & center from homography.
  4. Whether projected center lands in black padding.
  5. Rotation experiments: no rotation vs heading rotation vs local heading sweep.
  6. Dual homography evaluation: RANSAC vs MAGSAC vs DLT, shape scoring.
  7. pixel_to_latlon validation for both metatile and single-tile paths.

Outputs:
  outputs/phase_a/summary.csv           — one row per frame
  outputs/phase_a/per_tile_matches.csv  — one row per (frame, tile)
  outputs/phase_a/frame_NNN.json        — full detail per frame
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

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

NUM_TEST_FRAMES = 10          # Spread across 300-frame range
SEARCH_RADIUS_M = 500.0
RANSAC_THRESH = config.RANSAC_REPROJ_THRESH  # 8.0
HEADING_SWEEP_RANGE = range(-90, 91, 15)     # degrees around heading
OUTPUT_DIR = config.OUTPUT_DIR / "phase_a"


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg (CCW positive) around center, expanding canvas."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image, M, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def compute_shape_confidence(H: np.ndarray, img_w: int, img_h: int) -> Dict:
    """
    Compute FVL-SAR-style shape confidence from homography.
    Projects 4 corners of the query image through H and evaluates
    opposing-side, width-height, right-angle, and area consistency.

    Returns dict with individual terms and combined CShape.
    """
    corners = np.array([
        [0, 0], [img_w, 0], [img_w, img_h], [0, img_h]
    ], dtype=np.float64)
    proj = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H)[0]

    # Side lengths
    sides = []
    for i in range(4):
        p1 = proj[i]
        p2 = proj[(i + 1) % 4]
        sides.append(np.linalg.norm(p2 - p1))

    # Opposing side consistency: ratio of opposite sides
    opp_1 = min(sides[0], sides[2]) / max(sides[0], sides[2]) if max(sides[0], sides[2]) > 0 else 0
    opp_2 = min(sides[1], sides[3]) / max(sides[1], sides[3]) if max(sides[1], sides[3]) > 0 else 0
    opposing_side = (opp_1 + opp_2) / 2.0

    # Width-height consistency: projected vs original aspect ratio
    proj_w = (sides[0] + sides[2]) / 2.0
    proj_h = (sides[1] + sides[3]) / 2.0
    orig_aspect = img_w / img_h
    proj_aspect = proj_w / proj_h if proj_h > 0 else 0
    wh_ratio = min(orig_aspect, proj_aspect) / max(orig_aspect, proj_aspect) if max(orig_aspect, proj_aspect) > 0 else 0

    # Right-angle consistency: angles at corners
    angle_scores = []
    for i in range(4):
        v1 = proj[(i - 1) % 4] - proj[i]
        v2 = proj[(i + 1) % 4] - proj[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = np.degrees(np.arccos(abs(cos_angle)))
        angle_scores.append(1.0 - abs(angle_deg - 90) / 90.0)
    right_angle = np.mean(angle_scores)

    # Area consistency: projected area vs original area
    # Shoelace formula
    proj_area = 0.5 * abs(
        sum(proj[i][0] * proj[(i+1)%4][1] - proj[(i+1)%4][0] * proj[i][1] for i in range(4))
    )
    orig_area = img_w * img_h
    area_ratio = min(proj_area, orig_area) / max(proj_area, orig_area) if max(proj_area, orig_area) > 0 else 0

    # Convexity check
    cross_products = []
    for i in range(4):
        v1 = proj[(i+1)%4] - proj[i]
        v2 = proj[(i+2)%4] - proj[(i+1)%4]
        cross_products.append(v1[0]*v2[1] - v1[1]*v2[0])
    is_convex = all(c > 0 for c in cross_products) or all(c < 0 for c in cross_products)

    terms = [opposing_side, wh_ratio, right_angle, area_ratio]
    c_shape = 0.6 * min(terms) + 0.4 * np.mean(terms)

    return {
        "opposing_side": float(opposing_side),
        "width_height": float(wh_ratio),
        "right_angle": float(right_angle),
        "area": float(area_ratio),
        "c_shape": float(c_shape),
        "is_convex": bool(is_convex),
        "proj_corners": proj.tolist(),
        "proj_area": float(proj_area),
    }


def compute_homography_triple(kpts_q, kpts_r, matches, img_w, img_h, ransac_thresh=8.0):
    """
    Compute three homography variants:
      1. cv2.RANSAC (current pipeline)
      2. cv2.USAC_MAGSAC
      3. DLT (method=0, all points)

    Returns dict with results for each method.
    """
    if len(matches) < 4:
        return {"ransac": None, "magsac": None, "dlt": None}

    src = kpts_q[matches[:, 0]].astype(np.float64)
    dst = kpts_r[matches[:, 1]].astype(np.float64)

    results = {}

    # --- RANSAC ---
    try:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is not None:
            inliers = int(mask.ravel().sum())
            shape = compute_shape_confidence(H, img_w, img_h)
            # Reprojection error on inliers
            inlier_src = src[mask.ravel().astype(bool)]
            inlier_dst = dst[mask.ravel().astype(bool)]
            if len(inlier_src) > 0:
                proj = cv2.perspectiveTransform(inlier_src.reshape(1, -1, 2), H)[0]
                reproj_err = np.mean(np.linalg.norm(proj - inlier_dst, axis=1))
            else:
                reproj_err = float('inf')
            results["ransac"] = {
                "H": H.tolist(), "inliers": inliers, "total": len(matches),
                "reproj_error": float(reproj_err), **shape
            }
        else:
            results["ransac"] = None
    except Exception as e:
        results["ransac"] = {"error": str(e)}

    # --- MAGSAC ---
    try:
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh)
        if H is not None:
            inliers = int(mask.ravel().sum())
            shape = compute_shape_confidence(H, img_w, img_h)
            inlier_src = src[mask.ravel().astype(bool)]
            inlier_dst = dst[mask.ravel().astype(bool)]
            if len(inlier_src) > 0:
                proj = cv2.perspectiveTransform(inlier_src.reshape(1, -1, 2), H)[0]
                reproj_err = np.mean(np.linalg.norm(proj - inlier_dst, axis=1))
            else:
                reproj_err = float('inf')
            results["magsac"] = {
                "H": H.tolist(), "inliers": inliers, "total": len(matches),
                "reproj_error": float(reproj_err), **shape
            }
        else:
            results["magsac"] = None
    except Exception as e:
        results["magsac"] = {"error": str(e)}

    # --- DLT (all points, no outlier rejection) ---
    try:
        H, _ = cv2.findHomography(src, dst, 0)  # method=0 → DLT
        if H is not None:
            shape = compute_shape_confidence(H, img_w, img_h)
            # DLT uses all points, compute mean reproj error on all
            proj = cv2.perspectiveTransform(src.reshape(1, -1, 2), H)[0]
            reproj_err = np.mean(np.linalg.norm(proj - dst, axis=1))
            results["dlt"] = {
                "H": H.tolist(), "inliers": len(matches), "total": len(matches),
                "reproj_error": float(reproj_err), **shape
            }
        else:
            results["dlt"] = None
    except Exception as e:
        results["dlt"] = {"error": str(e)}

    return results


def check_pixel_in_black(meta_tile: np.ndarray, px_x: float, px_y: float,
                         threshold: int = 5) -> bool:
    """Check if pixel location falls in black (empty) region of meta-tile."""
    h, w = meta_tile.shape[:2]
    ix, iy = int(round(px_x)), int(round(px_y))
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return True  # outside canvas entirely
    # Check a 5x5 neighborhood
    region = meta_tile[max(0, iy-2):iy+3, max(0, ix-2):ix+3]
    return float(region.mean()) < threshold


def project_center(H: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    """Project image center through homography."""
    pt = np.array([[[w/2.0, h/2.0]]], dtype=np.float64)
    proj = cv2.perspectiveTransform(pt, H)[0][0]
    return float(proj[0]), float(proj[1])


def validate_pixel_to_latlon(tile_x: int, tile_y: int, zoom: int = 16, tile_px: int = 512):
    """
    Validate pixel_to_latlon for a single tile by checking:
    - pixel (0,0) = which corner?
    - pixel (256,256) = tile center?
    - pixel (0,512) = which corner?
    - pixel (512,0) = which corner?
    Compare against known tile_to_latlon values.
    """
    # Tile corners in latlon
    sw_lat, sw_lon = tile_to_latlon(tile_x, tile_y, zoom)         # SW corner
    ne_lat, ne_lon = tile_to_latlon(tile_x + 1, tile_y + 1, zoom) # NE corner
    nw_lat, nw_lon = tile_to_latlon(tile_x, tile_y + 1, zoom)     # NW corner
    se_lat, se_lon = tile_to_latlon(tile_x + 1, tile_y, zoom)     # SE corner
    center_lat, center_lon = tile_to_latlon(tile_x + 0.5, tile_y + 0.5, zoom)

    # What pixel_to_latlon_single_tile returns for key pixels
    from src.position_estimator import pixel_to_latlon_single_tile

    corners = {
        "px_0_0":       pixel_to_latlon_single_tile(0, 0, tile_x, tile_y, tile_px, zoom),
        "px_512_0":     pixel_to_latlon_single_tile(tile_px, 0, tile_x, tile_y, tile_px, zoom),
        "px_0_512":     pixel_to_latlon_single_tile(0, tile_px, tile_x, tile_y, tile_px, zoom),
        "px_512_512":   pixel_to_latlon_single_tile(tile_px, tile_px, tile_x, tile_y, tile_px, zoom),
        "px_256_256":   pixel_to_latlon_single_tile(tile_px/2, tile_px/2, tile_x, tile_y, tile_px, zoom),
    }
    known = {
        "SW": (sw_lat, sw_lon),
        "NE": (ne_lat, ne_lon),
        "NW": (nw_lat, nw_lon),
        "SE": (se_lat, se_lon),
        "center": (center_lat, center_lon),
    }

    # Check errors: which pixel maps to which real corner
    results = {}
    for px_name, (px_lat, px_lon) in corners.items():
        best_match = None
        best_err = float('inf')
        for known_name, (k_lat, k_lon) in known.items():
            err = haversine_distance(px_lat, px_lon, k_lat, k_lon)
            if err < best_err:
                best_err = err
                best_match = known_name
        results[px_name] = {
            "lat": px_lat, "lon": px_lon,
            "closest_corner": best_match, "error_m": best_err,
        }
    return results, known


# ═══════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTIC PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE A DIAGNOSTICS — Instrumentation & Orientation Experiments")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/6] Loading IMU data and building frame list...")
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
    print(f"  START_ROW={START_ROW}, aligned {len(aligned)} frames")

    # Pick N evenly-spaced test frames
    indices = np.linspace(0, len(aligned) - 1, NUM_TEST_FRAMES, dtype=int)
    test_frames = [aligned[i] for i in indices]
    print(f"  Selected {len(test_frames)} test frames at indices: {indices.tolist()}")

    # --- Initialize matcher and tile loader ---
    print("\n[2/6] Initializing matcher and tile loader...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    matcher = initialize_matcher(device, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=config.TMS_ZOOM_LEVEL,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    print(f"  Matcher ready, {len(tile_loader.list_tiles())} tiles available")

    # --- Validate pixel_to_latlon ---
    print("\n[3/6] Validating pixel_to_latlon (single tile)...")
    test_tile_x, test_tile_y = 34500, 45033
    px_results, known_corners = validate_pixel_to_latlon(test_tile_x, test_tile_y)
    print(f"  Tile ({test_tile_x}, {test_tile_y}):")
    print(f"  Known corners: SW=({known_corners['SW'][0]:.6f}, {known_corners['SW'][1]:.6f}), "
          f"NE=({known_corners['NE'][0]:.6f}, {known_corners['NE'][1]:.6f})")
    for px_name, info in px_results.items():
        print(f"    {px_name:12s} -> ({info['lat']:.6f}, {info['lon']:.6f})  "
              f"closest={info['closest_corner']:6s}  err={info['error_m']:.1f}m")

    # Save validation
    with open(OUTPUT_DIR / "pixel_to_latlon_validation.json", "w") as f:
        json.dump({"tile": [test_tile_x, test_tile_y],
                   "pixel_results": px_results,
                   "known_corners": {k: list(v) for k, v in known_corners.items()}}, f, indent=2)

    # --- Per-frame diagnostics ---
    print(f"\n[4/6] Running per-frame diagnostics on {len(test_frames)} frames...")

    all_per_tile = []
    summary_rows = []

    for fi, (csv_idx, ts, frame_path) in enumerate(test_frames):
        print(f"\n--- Frame {fi}/{len(test_frames)-1}  ts={ts:.3f}  path={frame_path.name} ---")
        row = imu_log.iloc[csv_idx]
        query_frame = load_image(frame_path)
        qh, qw = query_frame.shape[:2]

        ekf_lat = row['latitude_est']
        ekf_lon = row['longitude_est']
        heading_deg = row['yaw_deg']
        gt_lat = row['gps_lat']
        gt_lon = row['gps_lon']
        ekf_error = haversine_distance(ekf_lat, ekf_lon, gt_lat, gt_lon)

        print(f"  EKF=({ekf_lat:.5f},{ekf_lon:.5f})  GT=({gt_lat:.5f},{gt_lon:.5f})  "
              f"EKF_err={ekf_error:.1f}m  heading={heading_deg:.1f}°")

        # 4a. Find all candidate tiles
        candidates = find_tiles_within_radius(
            ekf_lat, ekf_lon, SEARCH_RADIUS_M,
            zoom=config.TMS_ZOOM_LEVEL,
            x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
            y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
        )
        print(f"  Candidates: {len(candidates)} tiles in {SEARCH_RADIUS_M}m radius")

        # 4b. Match unrotated query vs each tile
        tile_matches_unrotated = []
        for tx, ty in candidates:
            tile_img = tile_loader.load_aerial(tx, ty)
            if tile_img is None:
                continue
            res = matcher.match(query_frame, tile_img)
            tile_matches_unrotated.append({
                "tx": tx, "ty": ty,
                "num_matches": res["num_matches"],
                "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
            })
            all_per_tile.append({
                "frame_idx": fi, "ts": ts, "tx": tx, "ty": ty,
                "rotation": "none",
                "num_matches": res["num_matches"],
                "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
            })

        tile_matches_unrotated.sort(key=lambda r: r["num_matches"], reverse=True)
        top1_unrot = tile_matches_unrotated[0] if tile_matches_unrotated else None
        print(f"  Unrotated top-1: {top1_unrot}")

        # 4c. Rotate query by heading and match vs same tiles
        rotation_angle = -heading_deg  # rotate to north-up
        query_rotated = rotate_image(query_frame, rotation_angle)
        rh, rw = query_rotated.shape[:2]

        tile_matches_rotated = []
        for tx, ty in candidates:
            tile_img = tile_loader.load_aerial(tx, ty)
            if tile_img is None:
                continue
            res = matcher.match(query_rotated, tile_img)
            tile_matches_rotated.append({
                "tx": tx, "ty": ty,
                "num_matches": res["num_matches"],
                "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
            })
            all_per_tile.append({
                "frame_idx": fi, "ts": ts, "tx": tx, "ty": ty,
                "rotation": f"heading_{heading_deg:.1f}",
                "num_matches": res["num_matches"],
                "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
            })

        tile_matches_rotated.sort(key=lambda r: r["num_matches"], reverse=True)
        top1_rot = tile_matches_rotated[0] if tile_matches_rotated else None
        print(f"  Rotated (by {rotation_angle:.1f}°) top-1: {top1_rot}")

        # 4d. Heading sweep: try ±90° in 15° steps
        sweep_results = []
        best_sweep_tile = (candidates[0] if candidates else (0, 0))
        # Use only top-1 unrotated tile for sweep (speed)
        if top1_unrot:
            sweep_tile_img = tile_loader.load_aerial(top1_unrot["tx"], top1_unrot["ty"])
            if sweep_tile_img is not None:
                for angle in HEADING_SWEEP_RANGE:
                    q_rot = rotate_image(query_frame, angle)
                    res = matcher.match(q_rot, sweep_tile_img)
                    sweep_results.append({
                        "angle": angle,
                        "num_matches": res["num_matches"],
                        "mean_score": float(res["match_scores"].mean()) if res["num_matches"] > 0 else 0,
                    })
        best_sweep = max(sweep_results, key=lambda r: r["num_matches"]) if sweep_results else None
        print(f"  Heading sweep best: {best_sweep}")

        # 4e. Build meta-tile from top-3 unrotated, then compute homography + black detection
        top3 = tile_matches_unrotated[:3]
        meta_tile = None
        occupancy = {}
        homo_results = {}
        center_in_black = None
        proj_center_px = None

        if top3:
            xs = [t["tx"] for t in top3]
            ys = [t["ty"] for t in top3]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cols = x_max - x_min + 1
            rows = y_max - y_min + 1
            tile_px = config.TMS_TILE_SIZE_PX

            canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
            filled_cells = 0
            total_cells = rows * cols
            for t in top3:
                img = tile_loader.load_aerial(t["tx"], t["ty"])
                if img is None:
                    continue
                col = t["tx"] - x_min
                r = t["ty"] - y_min
                canvas[r * tile_px:(r + 1) * tile_px,
                       col * tile_px:(col + 1) * tile_px] = img
                filled_cells += 1

            meta_tile = canvas
            occupancy = {
                "grid_rows": rows, "grid_cols": cols,
                "filled_cells": filled_cells, "total_cells": total_cells,
                "fill_ratio": filled_cells / total_cells,
                "top3_tiles": [(t["tx"], t["ty"], t["num_matches"]) for t in top3],
            }
            print(f"  Meta-tile: {rows}x{cols} grid, {filled_cells}/{total_cells} filled")

            # Match query vs meta-tile
            meta_match = matcher.match(query_frame, meta_tile)
            print(f"  Meta-tile match: {meta_match['num_matches']} matches")

            # Triple homography
            if meta_match["num_matches"] >= 4:
                homo_results = compute_homography_triple(
                    meta_match["keypoints1"], meta_match["keypoints2"],
                    meta_match["matches"], qw, qh, RANSAC_THRESH)

                for method_name in ["ransac", "magsac", "dlt"]:
                    hr = homo_results.get(method_name)
                    if hr and "H" in hr:
                        print(f"    {method_name:8s}: inliers={hr['inliers']:3d} "
                              f"reproj={hr['reproj_error']:.2f}px  "
                              f"c_shape={hr['c_shape']:.3f}  "
                              f"convex={hr['is_convex']}")

                # Projected center analysis (use RANSAC H for now)
                if homo_results.get("ransac") and "H" in homo_results["ransac"]:
                    H = np.array(homo_results["ransac"]["H"])
                    cx, cy = project_center(H, qw, qh)
                    proj_center_px = (cx, cy)
                    in_bounds = (0 <= cx < meta_tile.shape[1] and 0 <= cy < meta_tile.shape[0])
                    in_black = check_pixel_in_black(meta_tile, cx, cy) if in_bounds else True
                    center_in_black = in_black

                    print(f"  Projected center: ({cx:.1f}, {cy:.1f})  "
                          f"in_bounds={in_bounds}  in_black={in_black}")

                    # Geo-coordinate from projected center
                    if in_bounds:
                        from src.position_estimator import pixel_to_latlon_in_metatile
                        top3_tiles = [(t["tx"], t["ty"], t["num_matches"]) for t in top3]
                        homo_lat, homo_lon = pixel_to_latlon_in_metatile(
                            cx, cy, top3_tiles, tile_px, config.TMS_ZOOM_LEVEL)
                        homo_error = haversine_distance(homo_lat, homo_lon, gt_lat, gt_lon)
                        print(f"  Homography position: ({homo_lat:.5f},{homo_lon:.5f})  "
                              f"error={homo_error:.1f}m")
                    else:
                        homo_lat, homo_lon, homo_error = None, None, None
                else:
                    homo_lat, homo_lon, homo_error = None, None, None
            else:
                homo_lat, homo_lon, homo_error = None, None, None

        # 4f. Also do triple homography on rotated query vs meta-tile
        homo_rotated_results = {}
        if meta_tile is not None:
            meta_match_rot = matcher.match(query_rotated, meta_tile)
            print(f"  Rotated meta-tile match: {meta_match_rot['num_matches']} matches")
            if meta_match_rot["num_matches"] >= 4:
                homo_rotated_results = compute_homography_triple(
                    meta_match_rot["keypoints1"], meta_match_rot["keypoints2"],
                    meta_match_rot["matches"], rw, rh, RANSAC_THRESH)
                for method_name in ["ransac", "magsac", "dlt"]:
                    hr = homo_rotated_results.get(method_name)
                    if hr and "H" in hr:
                        print(f"    ROT {method_name:8s}: inliers={hr['inliers']:3d} "
                              f"reproj={hr['reproj_error']:.2f}px  "
                              f"c_shape={hr['c_shape']:.3f}  "
                              f"convex={hr['is_convex']}")

        # --- Compile frame summary ---
        frame_data = {
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
            "unrot_top1_tx": top1_unrot["tx"] if top1_unrot else None,
            "unrot_top1_ty": top1_unrot["ty"] if top1_unrot else None,
            "unrot_top1_matches": top1_unrot["num_matches"] if top1_unrot else 0,
            "unrot_sum_matches": sum(t["num_matches"] for t in tile_matches_unrotated),
            # Rotated
            "rot_top1_tx": top1_rot["tx"] if top1_rot else None,
            "rot_top1_ty": top1_rot["ty"] if top1_rot else None,
            "rot_top1_matches": top1_rot["num_matches"] if top1_rot else 0,
            "rot_sum_matches": sum(t["num_matches"] for t in tile_matches_rotated),
            # Heading sweep
            "sweep_best_angle": best_sweep["angle"] if best_sweep else None,
            "sweep_best_matches": best_sweep["num_matches"] if best_sweep else 0,
            # Meta-tile
            "meta_fill_ratio": occupancy.get("fill_ratio"),
            "meta_grid": f"{occupancy.get('grid_rows','?')}x{occupancy.get('grid_cols','?')}",
            # Projected center
            "proj_center_x": proj_center_px[0] if proj_center_px else None,
            "proj_center_y": proj_center_px[1] if proj_center_px else None,
            "center_in_black": center_in_black,
            # Homography position
            "homo_lat": homo_lat, "homo_lon": homo_lon,
            "homo_error_m": homo_error,
        }

        # Add per-method homography stats
        for prefix, hresults in [("unrot", homo_results), ("rot", homo_rotated_results)]:
            for method in ["ransac", "magsac", "dlt"]:
                hr = hresults.get(method)
                if hr and isinstance(hr, dict) and "inliers" in hr:
                    frame_data[f"{prefix}_{method}_inliers"] = hr["inliers"]
                    frame_data[f"{prefix}_{method}_reproj"] = hr["reproj_error"]
                    frame_data[f"{prefix}_{method}_cshape"] = hr["c_shape"]
                    frame_data[f"{prefix}_{method}_convex"] = hr["is_convex"]
                else:
                    frame_data[f"{prefix}_{method}_inliers"] = None
                    frame_data[f"{prefix}_{method}_reproj"] = None
                    frame_data[f"{prefix}_{method}_cshape"] = None
                    frame_data[f"{prefix}_{method}_convex"] = None

        summary_rows.append(frame_data)

        # Save per-frame JSON
        detail = {
            **frame_data,
            "occupancy": occupancy,
            "tile_matches_unrotated": tile_matches_unrotated[:10],
            "tile_matches_rotated": tile_matches_rotated[:10],
            "sweep_results": sweep_results,
            "homo_triple_unrotated": {k: v for k, v in homo_results.items() if v is not None},
            "homo_triple_rotated": {k: v for k, v in homo_rotated_results.items() if v is not None},
            "pixel_validation": px_results if fi == 0 else "see frame_000.json",
        }
        with open(OUTPUT_DIR / f"frame_{fi:03d}.json", "w") as f:
            json.dump(detail, f, indent=2, default=str)

    # --- Save summary CSV ---
    print(f"\n[5/6] Saving summary CSV...")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"  Saved {len(summary_df)} rows to {OUTPUT_DIR / 'summary.csv'}")

    # Save per-tile CSV
    per_tile_df = pd.DataFrame(all_per_tile)
    per_tile_df.to_csv(OUTPUT_DIR / "per_tile_matches.csv", index=False)
    print(f"  Saved {len(per_tile_df)} rows to {OUTPUT_DIR / 'per_tile_matches.csv'}")

    # --- Print summary statistics ---
    print(f"\n[6/6] Summary Statistics")
    print("=" * 70)

    print(f"\n  EKF baseline error (test frames): "
          f"mean={summary_df['ekf_error_m'].mean():.1f}m, "
          f"median={summary_df['ekf_error_m'].median():.1f}m")

    print(f"\n  === Rotation Experiment ===")
    print(f"  Unrotated top-1 matches:  "
          f"mean={summary_df['unrot_top1_matches'].mean():.1f}, "
          f"min={summary_df['unrot_top1_matches'].min()}, "
          f"max={summary_df['unrot_top1_matches'].max()}")
    print(f"  Rotated   top-1 matches:  "
          f"mean={summary_df['rot_top1_matches'].mean():.1f}, "
          f"min={summary_df['rot_top1_matches'].min()}, "
          f"max={summary_df['rot_top1_matches'].max()}")
    delta = summary_df['rot_top1_matches'] - summary_df['unrot_top1_matches']
    print(f"  Delta (rot-unrot):        "
          f"mean={delta.mean():+.1f}, "
          f"min={delta.min():+d}, "
          f"max={delta.max():+d}")
    improved = (delta > 0).sum()
    print(f"  Frames improved by rotation: {improved}/{len(delta)}")

    print(f"\n  === Heading Sweep ===")
    print(f"  Best sweep angle:  {summary_df['sweep_best_angle'].tolist()}")
    print(f"  Best sweep matches:{summary_df['sweep_best_matches'].tolist()}")

    print(f"\n  === Meta-tile Occupancy ===")
    print(f"  Fill ratios: {summary_df['meta_fill_ratio'].tolist()}")
    print(f"  Center in black: {summary_df['center_in_black'].tolist()}")

    print(f"\n  === Homography Methods (unrotated) ===")
    for method in ["ransac", "magsac", "dlt"]:
        col_inl = f"unrot_{method}_inliers"
        col_rep = f"unrot_{method}_reproj"
        col_cs = f"unrot_{method}_cshape"
        col_cvx = f"unrot_{method}_convex"
        valid = summary_df[col_inl].dropna()
        if len(valid) > 0:
            print(f"  {method:8s}: inliers mean={valid.mean():.1f}  "
                  f"reproj mean={summary_df[col_rep].dropna().mean():.2f}px  "
                  f"cshape mean={summary_df[col_cs].dropna().mean():.3f}  "
                  f"convex={summary_df[col_cvx].dropna().sum():.0f}/{len(valid)}")

    print(f"\n  === Homography Methods (rotated) ===")
    for method in ["ransac", "magsac", "dlt"]:
        col_inl = f"rot_{method}_inliers"
        col_rep = f"rot_{method}_reproj"
        col_cs = f"rot_{method}_cshape"
        col_cvx = f"rot_{method}_convex"
        valid = summary_df[col_inl].dropna()
        if len(valid) > 0:
            print(f"  {method:8s}: inliers mean={valid.mean():.1f}  "
                  f"reproj mean={summary_df[col_rep].dropna().mean():.2f}px  "
                  f"cshape mean={summary_df[col_cs].dropna().mean():.3f}  "
                  f"convex={summary_df[col_cvx].dropna().sum():.0f}/{len(valid)}")

    if summary_df['homo_error_m'].dropna().any():
        print(f"\n  === Homography Position Error ===")
        valid_homo = summary_df['homo_error_m'].dropna()
        print(f"  Frames with valid homo: {len(valid_homo)}/{len(summary_df)}")
        print(f"  Mean error: {valid_homo.mean():.1f}m")
        print(f"  Per-frame: {valid_homo.tolist()}")

    print(f"\n  === pixel_to_latlon Validation ===")
    for px_name, info in px_results.items():
        flag = "OK" if info['error_m'] < 1.0 else f"ERROR {info['error_m']:.0f}m"
        print(f"    {px_name:12s} -> {info['closest_corner']:6s}  {flag}")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print("Phase A complete.")


if __name__ == "__main__":
    main()
