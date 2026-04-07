"""
Module 10 — Meta-Tile Builder.

Two-pass SuperPoint+LightGlue search, 8-neighbour expansion,
meta-tile construction, timestamped persistence, and meta-tile verification.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from src.tile_utils import TileLoader, find_tiles_within_radius

logger = logging.getLogger(__name__)

# 8-connected neighbour offsets (dx, dy)
NEIGHBOUR_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


class MetaTileBuilder:
    """
    Orchestrates:
      1. First-pass tile search (all tiles in radius)
      2. Second-pass 8-neighbour search around top-1
      3. Meta-tile construction from top-K
      4. Meta-tile persistence (always save before verification)
      5. Meta-tile verification against query frame
    """

    def __init__(self, feature_matcher, tile_loader: TileLoader, config):
        self.matcher = feature_matcher
        self.tiles = tile_loader
        self.cfg = config

    # ─── 10.2  First pass ────────────────────────────────────────

    def first_pass(self, query_frame: np.ndarray,
                   imu_lat: float, imu_lon: float,
                   search_radius_m: float
                   ) -> List[Tuple[int, int, int]]:
        """
        Match query against every tile within *search_radius_m*.

        Returns:
            Ranked list of (tile_x, tile_y, match_count) descending.
        """
        candidates = find_tiles_within_radius(
            imu_lat, imu_lon, search_radius_m,
            zoom=self.cfg.TMS_ZOOM_LEVEL,
            x_range=(self.cfg.TILE_X_MIN, self.cfg.TILE_X_MAX),
            y_range=(self.cfg.TILE_Y_MIN, self.cfg.TILE_Y_MAX),
        )

        results = []
        for tx, ty in candidates:
            tile_img = self.tiles.load_aerial(tx, ty)
            if tile_img is None:
                continue
            match_res = self.matcher.match(query_frame, tile_img)
            results.append((tx, ty, match_res["num_matches"]))

        results.sort(key=lambda r: r[2], reverse=True)
        return results

    # ─── 10.3  Second pass (8-neighbours) ────────────────────────

    def second_pass(self, query_frame: np.ndarray,
                    top_tile_x: int, top_tile_y: int
                    ) -> List[Tuple[int, int, int]]:
        """
        Match query against top-1 tile + its 8 grid neighbours.

        Returns:
            Ranked list of (tile_x, tile_y, match_count) descending.
        """
        neighbours = self._get_neighbours(top_tile_x, top_tile_y)
        results = []
        for tx, ty in neighbours:
            tile_img = self.tiles.load_aerial(tx, ty)
            if tile_img is None:
                continue
            match_res = self.matcher.match(query_frame, tile_img)
            results.append((tx, ty, match_res["num_matches"]))

        results.sort(key=lambda r: r[2], reverse=True)
        return results

    def _get_neighbours(self, tx: int, ty: int) -> List[Tuple[int, int]]:
        """Return pivot tile + existing 8-neighbours."""
        candidates = [(tx, ty)]
        for dx, dy in NEIGHBOUR_OFFSETS:
            nx, ny = tx + dx, ty + dy
            if self.tiles.exists(nx, ny):
                candidates.append((nx, ny))
        return candidates

    # ─── 10.4  Build meta-tile ───────────────────────────────────

    def build_meta_tile(self, top3_tiles: List[Tuple[int, int, int]]
                        ) -> np.ndarray:
        """
        Stitch top-K tiles into a single composite image.
        Black padding fills any empty grid cells.

        Returns:
            (H, W, 3) uint8 RGB canvas.
        """
        xs = [t[0] for t in top3_tiles]
        ys = [t[1] for t in top3_tiles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        tile_px = self.cfg.TMS_TILE_SIZE_PX

        canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
        for tx, ty, _ in top3_tiles:
            img = self.tiles.load_aerial(tx, ty)
            if img is None:
                continue
            col = tx - x_min
            row = ty - y_min
            canvas[row * tile_px:(row + 1) * tile_px,
                   col * tile_px:(col + 1) * tile_px] = img
        return canvas

    # ─── 10.5  Save meta-tile ────────────────────────────────────

    def save_meta_tile(self, meta_tile: np.ndarray,
                       query_timestamp: float) -> Path:
        """
        Persist meta-tile with timestamp-aligned filename.
        Always called *before* verification.
        """
        fname = f"metatile_{query_timestamp:.3f}.png"
        out_dir = Path(self.cfg.METATILE_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        Image.fromarray(meta_tile).save(out_path)
        return out_path

    # ─── 10.6  Verify meta-tile ──────────────────────────────────

    def verify_meta_tile(self, query_frame: np.ndarray,
                         meta_tile: np.ndarray) -> Tuple[int, Dict]:
        """
        Run matcher between query and meta-tile.

        Returns:
            (match_count, full_match_result_dict)
        """
        match_res = self.matcher.match(query_frame, meta_tile)
        return match_res["num_matches"], match_res

    # ─── 10.7  run (full orchestration) ──────────────────────────

    def run(self, query_frame: np.ndarray,
            imu_lat: float, imu_lon: float,
            query_timestamp: float,
            search_radius_m: Optional[float] = None) -> Optional[Dict]:
        """
        Execute the full two-pass pipeline:
        first pass → second pass → build → save → verify.

        Returns:
            dict with meta_tile, meta_tile_path, top3_tiles,
            verification_matches, verified, first_pass_candidates,
            match_result.
            None if no tiles found in first pass.
        """
        radius = search_radius_m or self.cfg.FIRST_PASS_SEARCH_RADIUS_M

        # Step 1 — first pass
        first_pass_results = self.first_pass(
            query_frame, imu_lat, imu_lon, radius)
        if not first_pass_results:
            logger.warning("MetaTileBuilder: no first-pass tiles for "
                           "(%.5f, %.5f) r=%.0f m", imu_lat, imu_lon, radius)
            return None

        # Step 2 — second pass on 8-neighbours of top-1
        top1_tx, top1_ty, _ = first_pass_results[0]
        second_pass_results = self.second_pass(
            query_frame, top1_tx, top1_ty)
        top_k = second_pass_results[:self.cfg.METATILE_TOP_K]

        # Step 3 — build meta-tile
        meta_tile = self.build_meta_tile(top_k)

        # Step 4 — save to disk (always, before verification)
        meta_tile_path = self.save_meta_tile(meta_tile, query_timestamp)

        # Step 5 — verify meta-tile against query
        match_count, match_result = self.verify_meta_tile(
            query_frame, meta_tile)
        verified = match_count >= self.cfg.METATILE_MATCH_THRESHOLD

        return {
            "meta_tile": meta_tile,
            "meta_tile_path": meta_tile_path,
            "top3_tiles": top_k,
            "verification_matches": match_count,
            "verified": verified,
            "first_pass_candidates": len(first_pass_results),
            "match_result": match_result,
        }
