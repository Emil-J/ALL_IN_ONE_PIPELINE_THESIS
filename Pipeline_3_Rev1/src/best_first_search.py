"""
Pipeline 1 — Best-First Search for Frame 0 (cold start).

Searches reference tiles within an IMU-defined radius, ranks by
SuperPoint+LightGlue match count, and returns the best localisation.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tile_utils import (
    TileLoader,
    find_tiles_within_radius,
    tile_to_latlon,
    haversine_distance,
)
from src.image_utils import preprocess_query_frame
from src.position_estimator import estimate_position, estimate_homography, query_center_in_reference, pixel_to_latlon_single_tile

logger = logging.getLogger(__name__)


class BestFirstSearcher:
    """
    Frame 0 cold-start searcher.

    1. Find candidate tiles within IMU search radius.
    2. Match query frame against each candidate with SuperPoint+LightGlue.
    3. Rank by match count, return top result with position estimate.
    """

    def __init__(self, feature_matcher, tile_loader: TileLoader, config):
        """
        Args:
            feature_matcher: SuperPointLightGlueMatcher instance.
            tile_loader: TileLoader for reference aerial tiles.
            config: config module with IMU_SEARCH_RADIUS_METERS, etc.
        """
        self.matcher = feature_matcher
        self.tiles = tile_loader
        self.cfg = config

    def search(self,
               query_frame: np.ndarray,
               imu_lat: float,
               imu_lon: float,
               search_radius_m: Optional[float] = None,
               ) -> Dict:
        """
        Run best-first search from IMU prior.

        Args:
            query_frame: Raw query frame (any size — will be preprocessed).
            imu_lat, imu_lon: IMU position estimate.
            search_radius_m: Override default search radius.

        Returns:
            dict with keys: position, heading, score, tiles_tested,
            search_time, ranked_tiles, match_result.
            Returns None values for position/heading if no tiles found.
        """
        t0 = time.perf_counter()
        radius = search_radius_m or self.cfg.IMU_SEARCH_RADIUS_METERS

        # Pre-process query frame once
        query_processed = preprocess_query_frame(
            query_frame,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )

        # 1. Find candidate tiles
        candidates = find_tiles_within_radius(
            imu_lat, imu_lon, radius,
            zoom=self.cfg.TMS_ZOOM_LEVEL,
            x_range=(self.cfg.TILE_X_MIN, self.cfg.TILE_X_MAX),
            y_range=(self.cfg.TILE_Y_MIN, self.cfg.TILE_Y_MAX),
        )

        if not candidates:
            logger.warning("No candidate tiles within %.0f m of (%.5f, %.5f)",
                           radius, imu_lat, imu_lon)
            return self._empty_result(t0)

        # 2. Match against each candidate
        results: List[Tuple[int, int, int, Dict]] = []
        for tx, ty in candidates:
            tile_img = self.tiles.load_aerial(tx, ty)
            if tile_img is None:
                continue
            match_res = self.matcher.match(query_processed, tile_img)
            results.append((tx, ty, match_res["num_matches"], match_res))

        if not results:
            logger.warning("All candidate tiles failed to load")
            return self._empty_result(t0)

        # 3. Rank descending by match count
        results.sort(key=lambda r: r[2], reverse=True)
        best_tx, best_ty, best_count, best_match = results[0]

        # 4. Position estimate via homography on best tile
        position = None
        heading = None
        homo = estimate_homography(
            best_match["keypoints1"],
            best_match["keypoints2"],
            best_match["matches"],
            ransac_thresh=self.cfg.RANSAC_REPROJ_THRESH,
            min_matches=self.cfg.MIN_MATCHES_FOR_HOMOGRAPHY,
        )
        if homo is not None:
            H, inlier_mask = homo
            ref_x, ref_y = query_center_in_reference(
                H, self.cfg.QUERY_RESIZE_WIDTH, self.cfg.SEMANTIC_INPUT_SIZE,
            )
            lat, lon = pixel_to_latlon_single_tile(
                ref_x, ref_y, best_tx, best_ty,
                tile_px=self.cfg.TMS_TILE_SIZE_PX,
                zoom=self.cfg.TMS_ZOOM_LEVEL,
            )
            position = (lat, lon)

        elapsed = time.perf_counter() - t0

        ranked_tiles = [(tx, ty, cnt) for tx, ty, cnt, _ in results[:self.cfg.TOP_K_CANDIDATES]]

        return {
            "position": position,
            "heading": heading,
            "score": best_count,
            "tiles_tested": len(results),
            "search_time": elapsed,
            "best_tile": (best_tx, best_ty),
            "ranked_tiles": ranked_tiles,
            "match_result": best_match,
        }

    # ── helpers ───────────────────────────────────────────────

    def _empty_result(self, t0: float) -> Dict:
        return {
            "position": None,
            "heading": None,
            "score": 0,
            "tiles_tested": 0,
            "search_time": time.perf_counter() - t0,
            "best_tile": None,
            "ranked_tiles": [],
            "match_result": None,
        }
