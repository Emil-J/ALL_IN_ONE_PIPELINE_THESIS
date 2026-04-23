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

    def __init__(self, feature_matcher, tile_loader: TileLoader, config,
                 feature_store=None):
        self.matcher = feature_matcher
        self.tiles = tile_loader
        self.cfg = config
        self.feature_store = feature_store

    # ─── 10.2  First pass ────────────────────────────────────────

    def first_pass(self, query_frame: np.ndarray,
                   imu_lat: float, imu_lon: float,
                   search_radius_m: float,
                   query_feats=None,
                   query_semantic_map=None,
                   ) -> List[Tuple[int, int, int]]:
        """
        Match query against every tile within *search_radius_m*.

        Args:
            query_feats: Pre-extracted SuperPoint features from
                         matcher.extract_features().  Extracted here
                         if not provided (for backward compatibility).
            query_semantic_map: Optional class mask (H, W) uint8.  When
                         provided, tiles are pre-filtered by semantic
                         histogram similarity before SuperPoint matching.

        Returns:
            Ranked list of (tile_x, tile_y, match_count) descending.
        """
        from src.semantic_tile_scorer import SemanticTileScorer

        candidates = find_tiles_within_radius(
            imu_lat, imu_lon, search_radius_m,
            zoom=self.cfg.TMS_ZOOM_LEVEL,
            x_range=(self.cfg.TILE_X_MIN, self.cfg.TILE_X_MAX),
            y_range=(self.cfg.TILE_Y_MIN, self.cfg.TILE_Y_MAX),
        )

        # Semantic pre-filter: rank candidates by histogram intersection,
        # keep only the top-K for SuperPoint (much cheaper than running
        # SuperPoint on all ~24 tiles).
        n_before_filter = len(candidates)
        if (query_semantic_map is not None
                and getattr(self.cfg, 'SEMANTIC_PREFILTER_ENABLED', True)
                and len(candidates) > getattr(self.cfg, 'SEMANTIC_PREFILTER_TOP_K', 10)):
            top_k_sem = getattr(self.cfg, 'SEMANTIC_PREFILTER_TOP_K', 10)
            scored = SemanticTileScorer.score_tiles(
                query_semantic_map, candidates, self.tiles)
            candidates = [(tx, ty) for tx, ty, _ in scored[:top_k_sem]]
            logger.debug("Semantic pre-filter: %d → %d candidates",
                         len(scored), len(candidates))

        # Extract query features once if not already provided
        if query_feats is None:
            query_feats = self.matcher.extract_features(query_frame)

        results = []
        for tx, ty in candidates:
            # Use precomputed reference features if available
            if self.feature_store is not None and self.feature_store.has_tile(tx, ty):
                ref_feats = self.feature_store.get_features(tx, ty)
                if ref_feats is not None:
                    match_res = self.matcher.match_both_precomputed(query_feats, ref_feats)
                    results.append((tx, ty, match_res["num_matches"]))
                    continue
            # Fallback: load tile image and extract features at runtime
            tile_img = self.tiles.load_aerial(tx, ty)
            if tile_img is None:
                continue
            match_res = self.matcher.match_precomputed(query_feats, tile_img)
            results.append((tx, ty, match_res["num_matches"]))

        results.sort(key=lambda r: r[2], reverse=True)
        return results

    # ─── 10.3  Second pass (8-neighbours) ────────────────────────

    def second_pass(self, query_frame: np.ndarray,
                    top_tile_x: int, top_tile_y: int,
                    query_feats=None,
                    ) -> List[Tuple[int, int, int]]:
        """
        Match query against top-1 tile + its 8 grid neighbours.

        Args:
            query_feats: Pre-extracted SuperPoint features (reused from
                         first_pass).  Extracted here if not provided.

        Returns:
            Ranked list of (tile_x, tile_y, match_count) descending.
        """
        if query_feats is None:
            query_feats = self.matcher.extract_features(query_frame)

        neighbours = self._get_neighbours(top_tile_x, top_tile_y)
        results = []
        for tx, ty in neighbours:
            # Use precomputed reference features if available
            if self.feature_store is not None and self.feature_store.has_tile(tx, ty):
                ref_feats = self.feature_store.get_features(tx, ty)
                if ref_feats is not None:
                    match_res = self.matcher.match_both_precomputed(query_feats, ref_feats)
                    results.append((tx, ty, match_res["num_matches"]))
                    continue
            # Fallback: load tile image and extract features at runtime
            tile_img = self.tiles.load_aerial(tx, ty)
            if tile_img is None:
                continue
            match_res = self.matcher.match_precomputed(query_feats, tile_img)
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
            row = y_max - ty  # north-up: northernmost tile (y_max) at row 0
            canvas[row * tile_px:(row + 1) * tile_px,
                   col * tile_px:(col + 1) * tile_px] = img
        return canvas

    # ─── 10.4b Build prediction meta-tile ────────────────────────

    def build_prediction_meta_tile(
        self, top3_tiles: List[Tuple[int, int, int]]
    ) -> Optional[np.ndarray]:
        """
        Stitch prediction tiles corresponding to top-K aerial tiles.

        Uses pre-computed color-coded prediction PNGs from disk instead of
        re-running UNet++ segmentation.  This is faster and uses the
        higher-quality offline predictions.

        Returns:
            (H, W, 3) uint8 RGB prediction canvas, or None if no prediction
            tiles are available.
        """
        # Check if any prediction tiles exist for the top-K
        found_any = False
        for tx, ty, _ in top3_tiles:
            if self.tiles.load_prediction(tx, ty) is not None:
                found_any = True
                break
        if not found_any:
            return None

        xs = [t[0] for t in top3_tiles]
        ys = [t[1] for t in top3_tiles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        tile_px = self.cfg.TMS_TILE_SIZE_PX

        canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
        for tx, ty, _ in top3_tiles:
            pred_img = self.tiles.load_prediction(tx, ty)
            if pred_img is None:
                continue
            # Ensure 3-channel RGB
            if pred_img.ndim == 2:
                pred_img = np.stack([pred_img]*3, axis=-1)
            elif pred_img.shape[2] == 4:
                pred_img = pred_img[:, :, :3]
            col = tx - x_min
            row = y_max - ty  # north-up
            canvas[row * tile_px:(row + 1) * tile_px,
                   col * tile_px:(col + 1) * tile_px] = pred_img[:, :, :3]
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
                         meta_tile: np.ndarray,
                         query_feats=None) -> Tuple[int, Dict]:
        """
        Run matcher between query and meta-tile.

        When *query_feats* is supplied (pre-extracted SuperPoint features),
        only the meta-tile SP extraction is run — the query extraction is
        skipped (~100ms saved per frame).

        Returns:
            (match_count, full_match_result_dict)
        """
        if query_feats is not None:
            # Reuse already-extracted query features — only extract meta-tile SP
            match_res = self.matcher.match_precomputed(query_feats, meta_tile)
        else:
            # Fallback: full extraction on both sides
            match_res = self.matcher.match(query_frame, meta_tile)
        return match_res["num_matches"], match_res

    # ─── 10.7  run (full orchestration) ──────────────────────────

    def run(self, query_frame: np.ndarray,
            imu_lat: float, imu_lon: float,
            query_timestamp: float,
            search_radius_m: Optional[float] = None,
            query_semantic_map=None) -> Optional[Dict]:
        """
        Execute the full two-pass pipeline:
        first pass → second pass → build → save → verify.

        Args:
            query_semantic_map: Optional class mask (H,W) uint8.  When
                provided, the first pass pre-filters tiles by semantic
                histogram similarity before running SuperPoint.

        Returns:
            dict with meta_tile, meta_tile_path, top3_tiles,
            verification_matches, verified, first_pass_candidates,
            match_result, _timing.
            None if no tiles found in first pass.
        """
        radius = search_radius_m or self.cfg.FIRST_PASS_SEARCH_RADIUS_M

        # Extract query SuperPoint features ONCE here; reused across all
        # first-pass and second-pass tile matches (~33 tiles/frame).
        query_feats = self.matcher.extract_features(query_frame)

        # Step 1 — first pass
        first_pass_results = self.first_pass(
            query_frame, imu_lat, imu_lon, radius,
            query_feats=query_feats,
            query_semantic_map=query_semantic_map,
        )

        if not first_pass_results:
            logger.warning("MetaTileBuilder: no first-pass tiles for "
                           "(%.5f, %.5f) r=%.0f m", imu_lat, imu_lon, radius)
            return None

        # Step 2 — second pass on 8-neighbours of top-1
        top1_tx, top1_ty, _ = first_pass_results[0]
        second_pass_results = self.second_pass(
            query_frame, top1_tx, top1_ty, query_feats=query_feats)
        top_k = second_pass_results[:self.cfg.METATILE_TOP_K]

        # Step 3 — build meta-tile
        meta_tile = self.build_meta_tile(top_k)

        # Step 3b — build prediction meta-tile (pre-computed, no model inference)
        prediction_meta_tile = self.build_prediction_meta_tile(top_k)

        # Step 4 — save to disk (debug only)
        if getattr(self.cfg, 'DEBUG_SAVE_METATILES', False):
            meta_tile_path = self.save_meta_tile(meta_tile, query_timestamp)
        else:
            meta_tile_path = None

        # Step 5 — verify meta-tile against query
        # Pass pre-extracted query_feats so we don't re-run SuperPoint on query
        match_count, match_result = self.verify_meta_tile(
            query_frame, meta_tile, query_feats=query_feats)
        verified = match_count >= self.cfg.METATILE_MATCH_THRESHOLD

        return {
            "meta_tile": meta_tile,
            "prediction_meta_tile": prediction_meta_tile,
            "meta_tile_path": meta_tile_path,
            "top3_tiles": top_k,
            "verification_matches": match_count,
            "verified": verified,
            "first_pass_candidates": len(first_pass_results),
            "match_result": match_result,
        }
