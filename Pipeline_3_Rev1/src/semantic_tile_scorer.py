"""
Semantic Tile Scorer — histogram-intersection pre-filter.

Scores candidate tiles by comparing the class distribution of the query
semantic mask against the class distribution of each reference prediction
tile.  Histogram intersection is viewpoint-invariant (both views cover
the same terrain) and fast (~1ms/tile, vs ~60ms for SuperPoint).

Reference prediction tiles are color-coded RGB PNGs using the palette:
    0 waterbodies  rgb(  4,   4, 255)
    1 forest_trees rgb(  0, 167,   2)
    2 land         rgb(243, 255, 150)
    3 railway      rgb(193, 105,  53)
    4 roads        rgb(255,   0, 231)
    5 buildings    rgb(150, 150, 150)
"""

import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

# RGB → class-index lookup table (from legend.txt)
_COLOR_TO_CLASS = {
    (  4,   4, 255): 0,   # waterbodies
    (  0, 167,   2): 1,   # forest_trees
    (243, 255, 150): 2,   # land
    (193, 105,  53): 3,   # railway
    (255,   0, 231): 4,   # roads
    (150, 150, 150): 5,   # buildings
}
NUM_CLASSES = 6


def _rgb_to_class_mask(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) uint8 color-coded prediction PNG to a
    (H, W) uint8 class-index mask.

    Pixels that don't match any known color are mapped to the closest
    class by L1 distance in RGB space (handles minor JPEG artifacts if
    tiles were ever re-saved as lossy).
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Build lookup: flatten RGB values → class
    # Use vectorized nearest-neighbor for speed
    ref_colors = np.array(list(_COLOR_TO_CLASS.keys()), dtype=np.int32)  # (6, 3)
    ref_classes = np.array(list(_COLOR_TO_CLASS.values()), dtype=np.uint8)

    flat = rgb.reshape(-1, 3).astype(np.int32)   # (N, 3)
    # L1 distance to each reference color: (N, 6)
    dists = np.sum(np.abs(flat[:, None, :] - ref_colors[None, :, :]), axis=2)
    nearest = np.argmin(dists, axis=1).astype(np.uint8)
    # Map nearest index → class id (handles arbitrary order in ref_classes)
    mask = ref_classes[nearest].reshape(h, w)
    return mask


def _class_histogram(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Compute a normalized class-frequency histogram from a class mask.

    Returns:
        (num_classes,) float32 array that sums to 1.0.
    """
    total = mask.size
    hist = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        hist[c] = np.count_nonzero(mask == c)
    if total > 0:
        hist /= total
    return hist


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Histogram intersection similarity ∈ [0, 1].

    score = Σ_c  min(h1[c], h2[c])

    1.0 means identical class distribution; 0.0 means no overlap.
    """
    return float(np.sum(np.minimum(h1, h2)))


class SemanticTileScorer:
    """Scores tiles by semantic class-distribution similarity."""

    @staticmethod
    def score_tiles(
        query_semantic_map: np.ndarray,
        candidate_tiles: List[Tuple[int, int]],
        tile_loader,
    ) -> List[Tuple[int, int, float]]:
        """Score candidate tiles by histogram intersection with the query.

        Args:
            query_semantic_map: (H, W) uint8 class-index mask for the
                                 current query frame.
            candidate_tiles:    List of (tile_x, tile_y) pairs to score.
            tile_loader:        TileLoader with a load_prediction() method.

        Returns:
            List of (tile_x, tile_y, score) sorted descending by score.
            Tiles with no prediction file are given score 0.5 (neutral).
        """
        query_hist = _class_histogram(query_semantic_map)

        scored: List[Tuple[int, int, float]] = []
        for tx, ty in candidate_tiles:
            pred_img = tile_loader.load_prediction(tx, ty)
            if pred_img is None:
                scored.append((tx, ty, 0.5))
                continue

            # Decode RGB color-coded PNG → class mask
            if pred_img.ndim == 3 and pred_img.shape[2] >= 3:
                ref_mask = _rgb_to_class_mask(pred_img[:, :, :3])
            elif pred_img.ndim == 2:
                # Already a class-index mask (raw uint8)
                ref_mask = pred_img.astype(np.uint8)
            else:
                scored.append((tx, ty, 0.5))
                continue

            ref_hist = _class_histogram(ref_mask)
            score = _histogram_intersection(query_hist, ref_hist)
            scored.append((tx, ty, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored


def compute_histogram_confidence(
    query_semantic_map: np.ndarray,
    reference_rgb: np.ndarray,
) -> float:
    """Compute histogram intersection between a query class mask and a
    reference RGB prediction image.

    Used by SemanticConfirmer.confirm() as a drop-in replacement for the
    centroid-based approach.

    Args:
        query_semantic_map: (H, W) uint8 class-index mask.
        reference_rgb:      (H, W, 3) uint8 color-coded prediction PNG.

    Returns:
        float in [0, 1].
    """
    query_hist = _class_histogram(query_semantic_map)

    if reference_rgb.ndim == 3 and reference_rgb.shape[2] >= 3:
        ref_mask = _rgb_to_class_mask(reference_rgb[:, :, :3])
    elif reference_rgb.ndim == 2:
        ref_mask = reference_rgb.astype(np.uint8)
    else:
        return 0.5

    ref_hist = _class_histogram(ref_mask)
    return _histogram_intersection(query_hist, ref_hist)
