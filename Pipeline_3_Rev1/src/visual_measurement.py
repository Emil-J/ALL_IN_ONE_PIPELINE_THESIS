"""
Visual Measurement Module — Phase B1.

Implements:
  1. Query heading rotation (CCW rotation to align with north-up reference tiles)
  2. Dual homography evaluation (MAGSAC branch + DLT branch)
  3. Shape confidence scoring (FVL-SAR paper)
  4. Multiple visual measurement extraction methods
  5. Attitude-aware nadir correction
"""

import math
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any

from src.tile_utils import tile_to_latlon, haversine_distance


# ═══════════════════════════════════════════════════════════════
# 1. QUERY HEADING ROTATION
# ═══════════════════════════════════════════════════════════════

def rotate_image(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image CCW by angle_deg around its center, expanding the canvas
    so no content is lost. Returns the rotated image and the 2x3 affine
    transform matrix (for mapping original pixel coords to rotated coords).

    Args:
        image: (H, W, 3) uint8 array.
        angle_deg: rotation angle in degrees (positive = CCW).

    Returns:
        (rotated_image, M_affine) where M_affine is the 2x3 affine matrix.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new canvas size to fit rotated image
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(math.ceil(w * cos_a + h * sin_a))
    new_h = int(math.ceil(w * sin_a + h * cos_a))

    # Adjust translation to center the rotated image
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    return rotated, M


def get_rotation_inverse(M_fwd: np.ndarray) -> np.ndarray:
    """
    Get the inverse affine matrix for a forward rotation.
    Converts points in rotated space back to original space.
    """
    # Build 3x3 matrix, invert, take top 2 rows
    M3 = np.vstack([M_fwd, [0, 0, 1]])
    M3_inv = np.linalg.inv(M3)
    return M3_inv[:2, :]


# ═══════════════════════════════════════════════════════════════
# 2. SHAPE CONFIDENCE SCORING (FVL-SAR paper)
# ═══════════════════════════════════════════════════════════════

def compute_shape_confidence(H: np.ndarray, query_w: int, query_h: int
                             ) -> Dict[str, float]:
    """
    Compute the FVL-SAR shape confidence score from a homography.

    Projects the four query corners through H and evaluates:
      - opposing_side: min(side_ratio) of opposing edge pairs
      - width_height: min(w,h)/max(w,h) of projected quad
      - right_angle: penalty for non-90° angles
      - area: projected area vs expected area
      - convexity: whether the projected quad is convex

    CShape = 0.6 * min(terms) + 0.4 * mean(terms)

    Args:
        H: 3x3 homography (query -> reference).
        query_w, query_h: query image dimensions.

    Returns:
        dict with individual terms, CShape, and convexity.
    """
    corners = np.array([
        [0, 0],
        [query_w, 0],
        [query_w, query_h],
        [0, query_h],
    ], dtype=np.float64)

    # Project corners
    proj = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), H).reshape(-1, 2)

    # Edge lengths
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append(np.linalg.norm(proj[j] - proj[i]))

    if min(edges) < 1e-6:
        return {"opposing_side": 0, "width_height": 0, "right_angle": 0,
                "area": 0, "CShape": 0, "convex": False}

    # Opposing side ratio
    r1 = min(edges[0], edges[2]) / max(edges[0], edges[2])
    r2 = min(edges[1], edges[3]) / max(edges[1], edges[3])
    opposing_side = min(r1, r2)

    # Width/height ratio
    avg_w = (edges[0] + edges[2]) / 2
    avg_h = (edges[1] + edges[3]) / 2
    width_height = min(avg_w, avg_h) / max(avg_w, avg_h)

    # Right angle score
    angles = []
    for i in range(4):
        a = proj[(i - 1) % 4]
        b = proj[i]
        c = proj[(i + 1) % 4]
        v1 = a - b
        v2 = c - b
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)
    right_angle = 1.0 - max(abs(a - 90) for a in angles) / 90.0
    right_angle = max(0, right_angle)

    # Area ratio
    proj_area = cv2.contourArea(proj.astype(np.float32))
    expected_area = query_w * query_h
    area = min(proj_area, expected_area) / max(proj_area, expected_area) if expected_area > 0 else 0

    # Convexity test
    hull = cv2.convexHull(proj.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    convex = abs(hull_area - proj_area) < 0.01 * max(hull_area, 1.0)

    terms = [opposing_side, width_height, right_angle, area]
    CShape = 0.6 * min(terms) + 0.4 * np.mean(terms)

    return {
        "opposing_side": opposing_side,
        "width_height": width_height,
        "right_angle": right_angle,
        "area": area,
        "CShape": CShape,
        "convex": convex,
    }


# ═══════════════════════════════════════════════════════════════
# 3. DUAL HOMOGRAPHY EVALUATION
# ═══════════════════════════════════════════════════════════════

def compute_dual_homography(src_pts: np.ndarray, dst_pts: np.ndarray,
                            query_w: int, query_h: int,
                            ransac_thresh: float = 8.0
                            ) -> Dict[str, Any]:
    """
    Compute two explicit homography branches from the same correspondences:

    Branch A — DLT (Direct Linear Transform):
      cv2.findHomography with method=0 (no outlier rejection).
      This uses ALL correspondences to compute the least-squares homography.
      Good when most correspondences are correct; fragile to outliers.

    Branch B — MAGSAC++ (robust):
      cv2.findHomography with method=cv2.USAC_MAGSAC.
      Marginalized sample consensus — robust to outliers.
      Returns inlier mask.

    For each branch, computes:
      - H matrix
      - inlier count (for DLT: points within threshold; for MAGSAC: its own mask)
      - reprojection error
      - shape confidence (CShape)
      - convexity
      - projected query center

    Then selects the winner based on a composite score.

    Args:
        src_pts: (N, 2) query keypoint coords.
        dst_pts: (N, 2) reference keypoint coords.
        query_w, query_h: query image dimensions.
        ransac_thresh: reprojection threshold in pixels.

    Returns:
        dict with 'dlt', 'magsac', 'winner', and winner's results.
    """
    n = len(src_pts)
    result = {"n_correspondences": n, "dlt": None, "magsac": None,
              "winner": None, "winner_H": None, "winner_mask": None}

    if n < 4:
        return result

    src = src_pts.astype(np.float64)
    dst = dst_pts.astype(np.float64)

    # ── Branch A: DLT ──
    try:
        H_dlt, _ = cv2.findHomography(src, dst, method=0)
        if H_dlt is not None:
            # Compute reprojection for all points
            projected = cv2.perspectiveTransform(
                src.reshape(1, -1, 2), H_dlt).reshape(-1, 2)
            reproj_errors = np.linalg.norm(projected - dst, axis=1)
            dlt_inlier_mask = reproj_errors < ransac_thresh
            dlt_inliers = int(dlt_inlier_mask.sum())
            dlt_reproj = float(np.median(reproj_errors))
            dlt_shape = compute_shape_confidence(H_dlt, query_w, query_h)

            # Project query center
            cx, cy = query_w / 2.0, query_h / 2.0
            pt = np.array([[cx, cy]], dtype=np.float64)
            proj_center = cv2.perspectiveTransform(
                pt.reshape(1, 1, 2), H_dlt).reshape(2)

            result["dlt"] = {
                "H": H_dlt,
                "inliers": dlt_inliers,
                "inlier_mask": dlt_inlier_mask,
                "reproj_median": dlt_reproj,
                "reproj_all": reproj_errors,
                "CShape": dlt_shape["CShape"],
                "convex": dlt_shape["convex"],
                "shape_detail": dlt_shape,
                "proj_center": proj_center,
            }
    except cv2.error:
        pass

    # ── Branch B: MAGSAC++ ──
    try:
        H_mag, mask_mag = cv2.findHomography(
            src, dst, cv2.USAC_MAGSAC, ransac_thresh)
        if H_mag is not None:
            mask_bool = mask_mag.ravel().astype(bool)
            mag_inliers = int(mask_bool.sum())

            projected = cv2.perspectiveTransform(
                src.reshape(1, -1, 2), H_mag).reshape(-1, 2)
            reproj_errors = np.linalg.norm(projected - dst, axis=1)
            mag_reproj = float(np.median(reproj_errors[mask_bool])) if mag_inliers > 0 else 999

            mag_shape = compute_shape_confidence(H_mag, query_w, query_h)

            cx, cy = query_w / 2.0, query_h / 2.0
            pt = np.array([[cx, cy]], dtype=np.float64)
            proj_center = cv2.perspectiveTransform(
                pt.reshape(1, 1, 2), H_mag).reshape(2)

            result["magsac"] = {
                "H": H_mag,
                "inliers": mag_inliers,
                "inlier_mask": mask_bool,
                "reproj_median": mag_reproj,
                "reproj_all": reproj_errors,
                "CShape": mag_shape["CShape"],
                "convex": mag_shape["convex"],
                "shape_detail": mag_shape,
                "proj_center": proj_center,
            }
    except cv2.error:
        pass

    # ── Winner selection ──
    result["winner"], result["winner_H"], result["winner_mask"] = \
        _select_homography_winner(result["dlt"], result["magsac"])

    return result


def _select_homography_winner(dlt: Optional[Dict], magsac: Optional[Dict]
                              ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Select the better homography based on a composite score.

    Score = inlier_ratio * CShape * convexity_bonus
    where convexity_bonus = 1.5 if convex else 1.0

    MAGSAC wins ties (it's the more robust estimator).
    """
    if dlt is None and magsac is None:
        return None, None, None
    if dlt is None:
        return "magsac", magsac["H"], magsac["inlier_mask"]
    if magsac is None:
        return "dlt", dlt["H"], dlt["inlier_mask"]

    def score(branch: Dict) -> float:
        n = branch["inliers"]
        cs = branch["CShape"]
        bonus = 1.5 if branch["convex"] else 1.0
        return n * cs * bonus

    s_dlt = score(dlt)
    s_mag = score(magsac)

    if s_mag >= s_dlt:
        return "magsac", magsac["H"], magsac["inlier_mask"]
    else:
        return "dlt", dlt["H"], dlt["inlier_mask"]


# ═══════════════════════════════════════════════════════════════
# 4. VISUAL MEASUREMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_visual_measurements(
    H: np.ndarray,
    inlier_mask: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    query_w: int, query_h: int,
    top3_tiles: List[Tuple[int, int, int]],
    tile_px: int = 512,
    zoom: int = 16,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
    altitude_m: float = 365.76,  # ~1200ft default
) -> Dict[str, Any]:
    """
    Extract geographic measurements from a homography using multiple methods.

    Methods:
    A. projected_center: project query image center through H
    B. inlier_centroid_ref: centroid of inlier reference keypoints
    C. inlier_trimmed_centroid: trimmed mean (drop 10% extremes) of inlier ref keypoints
    D. nadir_corrected: shift query center by pitch/roll before projection
    E. weighted_inlier_centroid: inlier centroid weighted by inverse reproj error

    Args:
        H: 3x3 homography (query -> reference meta-tile)
        inlier_mask: boolean mask over correspondences
        src_pts: (N, 2) query keypoints
        dst_pts: (N, 2) reference keypoints (in meta-tile pixel space)
        query_w, query_h: query image size
        top3_tiles: tiles defining the meta-tile grid
        tile_px: tile pixel size
        zoom: TMS zoom level
        pitch_rad: aircraft pitch in radians (positive = nose up)
        roll_rad: aircraft roll in radians (positive = right wing down)
        altitude_m: altitude above ground in meters

    Returns:
        dict mapping method name -> {"px": (x,y), "latlon": (lat,lon), "valid": bool}
    """
    results = {}

    xs = [t[0] for t in top3_tiles]
    ys = [t[1] for t in top3_tiles]
    x_min, y_min = min(xs), min(ys)
    y_max = max(ys)
    meta_w = (max(xs) - x_min + 1) * tile_px
    meta_h = (y_max - y_min + 1) * tile_px

    def px_to_latlon(px_x: float, px_y: float) -> Tuple[float, float]:
        """Convert meta-tile pixel to lat/lon (north-up canvas)."""
        tile_x_frac = x_min + px_x / tile_px
        tile_y_frac = (y_max + 1) - px_y / tile_px
        return tile_to_latlon(tile_x_frac, tile_y_frac, zoom)

    def is_valid_px(px_x: float, px_y: float) -> bool:
        """Check if pixel is within meta-tile canvas bounds."""
        return 0 <= px_x <= meta_w and 0 <= px_y <= meta_h

    def check_in_black(px_x: float, px_y: float) -> bool:
        """Check if pixel falls in a cell not covered by any tile."""
        col = int(px_x // tile_px)
        row = int(px_y // tile_px)
        # Check if any tile occupies this grid cell (north-up: row = y_max - ty)
        for tx, ty, _ in top3_tiles:
            if (tx - x_min) == col and (y_max - ty) == row:
                return False
        return True  # no tile covers this cell = black

    # ── Method A: Projected query center ──
    cx, cy = query_w / 2.0, query_h / 2.0
    pt = np.array([[cx, cy]], dtype=np.float64)
    proj_center = cv2.perspectiveTransform(pt.reshape(1, 1, 2), H).reshape(2)
    pc_x, pc_y = float(proj_center[0]), float(proj_center[1])
    pc_valid = is_valid_px(pc_x, pc_y) and not check_in_black(pc_x, pc_y)
    if is_valid_px(pc_x, pc_y):
        pc_latlon = px_to_latlon(pc_x, pc_y)
    else:
        pc_latlon = (None, None)
    results["projected_center"] = {"px": (pc_x, pc_y), "latlon": pc_latlon,
                                    "valid": pc_valid}

    # ── Inlier points in reference space ──
    inlier_dst = dst_pts[inlier_mask]

    # ── Method B: Inlier centroid in reference ──
    if len(inlier_dst) >= 4:
        centroid = np.mean(inlier_dst, axis=0)
        ic_x, ic_y = float(centroid[0]), float(centroid[1])
        ic_valid = is_valid_px(ic_x, ic_y) and not check_in_black(ic_x, ic_y)
        if is_valid_px(ic_x, ic_y):
            ic_latlon = px_to_latlon(ic_x, ic_y)
        else:
            ic_latlon = (None, None)
        results["inlier_centroid"] = {"px": (ic_x, ic_y), "latlon": ic_latlon,
                                       "valid": ic_valid}
    else:
        results["inlier_centroid"] = {"px": (None, None), "latlon": (None, None),
                                       "valid": False}

    # ── Method C: Trimmed inlier centroid (drop 10% extremes) ──
    if len(inlier_dst) >= 10:
        # Compute distance from raw centroid, drop furthest 10%
        raw_centroid = np.mean(inlier_dst, axis=0)
        dists = np.linalg.norm(inlier_dst - raw_centroid, axis=1)
        cutoff = np.percentile(dists, 90)
        trimmed = inlier_dst[dists <= cutoff]
        if len(trimmed) >= 4:
            tc = np.mean(trimmed, axis=0)
            tc_x, tc_y = float(tc[0]), float(tc[1])
            tc_valid = is_valid_px(tc_x, tc_y) and not check_in_black(tc_x, tc_y)
            if is_valid_px(tc_x, tc_y):
                tc_latlon = px_to_latlon(tc_x, tc_y)
            else:
                tc_latlon = (None, None)
            results["trimmed_centroid"] = {"px": (tc_x, tc_y), "latlon": tc_latlon,
                                            "valid": tc_valid}
        else:
            results["trimmed_centroid"] = {"px": (None, None), "latlon": (None, None),
                                            "valid": False}
    else:
        results["trimmed_centroid"] = {"px": (None, None), "latlon": (None, None),
                                        "valid": False}

    # ── Method D: Nadir-corrected projection ──
    # Near-nadir, the camera optical axis points approximately at image center.
    # With non-zero pitch/roll, the nadir point shifts:
    #   dx = altitude * tan(roll)   (positive roll = right wing down = nadir shifts left)
    #   dy = altitude * tan(pitch)  (positive pitch = nose up = nadir shifts forward/up)
    # Convert from meters to pixels using focal length approximation.
    # For a typical drone camera with ~70° FOV at 1920px width:
    #   f_px ~ w / (2 * tan(FOV/2)) ~ 1920 / (2 * tan(35°)) ~ 1371 px
    # The nadir ground point in query image coordinates shifts by:
    #   shift_x = -f_px * tan(roll)   (shift toward low wing)
    #   shift_y = -f_px * tan(pitch)  (shift behind nose)
    f_px_approx = query_w / (2 * math.tan(math.radians(35)))
    nadir_x = cx - f_px_approx * math.tan(roll_rad)
    nadir_y = cy - f_px_approx * math.tan(pitch_rad)

    nadir_pt = np.array([[nadir_x, nadir_y]], dtype=np.float64)
    nadir_proj = cv2.perspectiveTransform(nadir_pt.reshape(1, 1, 2), H).reshape(2)
    np_x, np_y = float(nadir_proj[0]), float(nadir_proj[1])
    np_valid = is_valid_px(np_x, np_y) and not check_in_black(np_x, np_y)
    if is_valid_px(np_x, np_y):
        np_latlon = px_to_latlon(np_x, np_y)
    else:
        np_latlon = (None, None)
    results["nadir_corrected"] = {"px": (np_x, np_y), "latlon": np_latlon,
                                   "valid": np_valid,
                                   "pitch_rad": pitch_rad, "roll_rad": roll_rad,
                                   "nadir_shift_px": (nadir_x - cx, nadir_y - cy)}

    # ── Method E: Weighted inlier centroid ──
    if len(inlier_dst) >= 4:
        inlier_src = src_pts[inlier_mask]
        projected = cv2.perspectiveTransform(
            inlier_src.reshape(1, -1, 2).astype(np.float64), H).reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - inlier_dst, axis=1)
        # Weight = 1/(reproj_error + epsilon)
        weights = 1.0 / (reproj_errors + 0.5)
        weights /= weights.sum()
        weighted_centroid = np.average(inlier_dst, weights=weights, axis=0)
        wc_x, wc_y = float(weighted_centroid[0]), float(weighted_centroid[1])
        wc_valid = is_valid_px(wc_x, wc_y) and not check_in_black(wc_x, wc_y)
        if is_valid_px(wc_x, wc_y):
            wc_latlon = px_to_latlon(wc_x, wc_y)
        else:
            wc_latlon = (None, None)
        results["weighted_centroid"] = {"px": (wc_x, wc_y), "latlon": wc_latlon,
                                         "valid": wc_valid}
    else:
        results["weighted_centroid"] = {"px": (None, None), "latlon": (None, None),
                                         "valid": False}

    return results


# ═══════════════════════════════════════════════════════════════
# 5. FULL VISUAL MEASUREMENT PIPELINE
# ═══════════════════════════════════════════════════════════════

def compute_visual_measurement(
    match_result: Dict,
    top3_tiles: List[Tuple[int, int, int]],
    query_w: int, query_h: int,
    tile_px: int = 512,
    zoom: int = 16,
    ransac_thresh: float = 8.0,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
    altitude_m: float = 365.76,
) -> Optional[Dict[str, Any]]:
    """
    Complete pipeline: match result -> dual homography -> measurement extraction.

    Returns None if insufficient matches.
    """
    matches = match_result["matches"]
    if len(matches) < 4:
        return None

    src_pts = match_result["keypoints1"][matches[:, 0]]
    dst_pts = match_result["keypoints2"][matches[:, 1]]

    # Dual homography
    dual = compute_dual_homography(src_pts, dst_pts, query_w, query_h, ransac_thresh)

    if dual["winner"] is None:
        return None

    H = dual["winner_H"]
    mask = dual["winner_mask"]

    # Extract measurements
    measurements = extract_visual_measurements(
        H, mask, src_pts, dst_pts,
        query_w, query_h, top3_tiles,
        tile_px=tile_px, zoom=zoom,
        pitch_rad=pitch_rad, roll_rad=roll_rad,
        altitude_m=altitude_m,
    )

    return {
        "dual_homography": dual,
        "measurements": measurements,
        "winner": dual["winner"],
        "n_matches": len(matches),
    }
