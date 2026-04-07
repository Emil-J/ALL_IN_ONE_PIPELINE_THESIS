"""
Position estimator — derive GPS coordinates from homography between
query frame and geo-referenced meta-tile / individual tile.
"""

import math
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict

from src.tile_utils import tile_to_latlon, tile_bounds


def estimate_homography(kpts_query: np.ndarray,
                        kpts_ref: np.ndarray,
                        matches: np.ndarray,
                        ransac_thresh: float = 8.0,
                        min_matches: int = 4
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate homography from matched keypoint pairs.

    Args:
        kpts_query: (N, 2) keypoints in query image.
        kpts_ref:   (M, 2) keypoints in reference image.
        matches:    (K, 2) index pairs into kpts_query / kpts_ref.
        ransac_thresh: RANSAC reprojection threshold in pixels.
        min_matches: minimum matches required.

    Returns:
        (H, inlier_mask) or None if not enough matches.
    """
    if len(matches) < min_matches:
        return None

    src_pts = kpts_query[matches[:, 0]].astype(np.float64)
    dst_pts = kpts_ref[matches[:, 1]].astype(np.float64)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    return H, mask.ravel().astype(bool)


def query_center_in_reference(H: np.ndarray,
                              query_w: int,
                              query_h: int) -> Tuple[float, float]:
    """
    Project the centre of the query frame into reference-image pixel coords
    using homography H (query → reference).

    Returns:
        (ref_x, ref_y) in pixels.
    """
    cx, cy = query_w / 2.0, query_h / 2.0
    pt = np.array([cx, cy, 1.0])
    proj = H @ pt
    proj /= proj[2]
    return float(proj[0]), float(proj[1])


def pixel_to_latlon_in_metatile(ref_px_x: float,
                                ref_px_y: float,
                                top3_tiles: List[Tuple[int, int, int]],
                                tile_px: int = 512,
                                zoom: int = 16) -> Tuple[float, float]:
    """
    Convert a pixel position inside a meta-tile canvas to (lat, lon).

    The meta-tile canvas is built by `MetaTileBuilder.build_meta_tile` which
    lays tiles on a grid whose origin is (min_tile_x, min_tile_y).

    Args:
        ref_px_x, ref_px_y: pixel coords in the meta-tile image.
        top3_tiles: list of (tile_x, tile_y, score) used to build the meta-tile.
        tile_px: pixel size of one tile (512).
        zoom: TMS zoom level.

    Returns:
        (latitude, longitude)
    """
    xs = [t[0] for t in top3_tiles]
    ys = [t[1] for t in top3_tiles]
    x_min = min(xs)
    y_min = min(ys)

    # Fractional tile position in TMS grid
    tile_x_frac = x_min + ref_px_x / tile_px
    tile_y_frac = y_min + ref_px_y / tile_px

    return tile_to_latlon(tile_x_frac, tile_y_frac, zoom)


def pixel_to_latlon_single_tile(ref_px_x: float,
                                ref_px_y: float,
                                tile_x: int,
                                tile_y: int,
                                tile_px: int = 512,
                                zoom: int = 16) -> Tuple[float, float]:
    """Convert pixel coords within a single tile to (lat, lon)."""
    tile_x_frac = tile_x + ref_px_x / tile_px
    tile_y_frac = tile_y + ref_px_y / tile_px
    return tile_to_latlon(tile_x_frac, tile_y_frac, zoom)


def estimate_position(match_result: Dict,
                      top3_tiles: List[Tuple[int, int, int]],
                      query_w: int = 512,
                      query_h: int = 512,
                      tile_px: int = 512,
                      zoom: int = 16,
                      ransac_thresh: float = 8.0,
                      min_matches: int = 4) -> Optional[Dict]:
    """
    Full pipeline: match result → GPS coordinate via homography.

    Args:
        match_result: output of SuperPointLightGlueMatcher.match() —
                      must contain keypoints1, keypoints2, matches.
        top3_tiles: tiles used to construct the meta-tile.
        query_w, query_h: query frame dimensions (after preprocessing).

    Returns:
        dict with lat, lon, inliers, H or None on failure.
    """
    homo = estimate_homography(
        match_result["keypoints1"],
        match_result["keypoints2"],
        match_result["matches"],
        ransac_thresh=ransac_thresh,
        min_matches=min_matches,
    )
    if homo is None:
        return None

    H, inlier_mask = homo
    ref_x, ref_y = query_center_in_reference(H, query_w, query_h)
    lat, lon = pixel_to_latlon_in_metatile(ref_x, ref_y, top3_tiles,
                                           tile_px=tile_px, zoom=zoom)

    return {
        "lat": lat,
        "lon": lon,
        "ref_px": (ref_x, ref_y),
        "inliers": int(inlier_mask.sum()),
        "H": H,
    }
