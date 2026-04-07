"""
Module 11 — Semantic Confirmer.

Histogram-intersection confirmation between query semantic map and
reference prediction tiles covering the meta-tile area.

Replaces the original centroid-based approach which broke due to spatial
mismatch between the oblique MSFS query and the orthophoto reference.
Class histograms are viewpoint-invariant: if you are over forest, both
the oblique and top-down views show ~80% forest pixels.
"""

import numpy as np
from typing import Dict, List, Optional

from src.image_utils import preprocess_query_frame
from src.semantic_tile_scorer import compute_histogram_confidence


class SemanticConfirmer:
    """
    Semantic confirmation via histogram intersection.

    confirm() segments the meta-tile prediction and compares class
    distributions with the query.  High score = flying over matching
    terrain type.
    """

    def __init__(self, semantic_model, config):
        self.model = semantic_model
        self.cfg = config

    # kept for backward-compatibility; no longer used for confirm()
    def segment(self, image: np.ndarray) -> np.ndarray:
        processed = preprocess_query_frame(
            image,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        return self.model.predict(processed)

    # ─── 11.5  Confirm ───────────────────────────────────────────

    def confirm(self, query_semantic_map: np.ndarray,
                meta_tile: np.ndarray) -> Dict:
        """
        Semantic confirmation by histogram intersection.

        Segments the meta-tile with the UNet++ model, then computes the
        histogram intersection between query and meta-tile class
        distributions.  Score ∈ [0, 1].

        Args:
            query_semantic_map: Pre-computed query class mask (H, W) uint8.
            meta_tile:          RGB meta-tile image (used for segmentation).

        Returns:
            dict with 'confidence' (float), 'query_hist', 'ref_hist'.
        """
        # Segment meta-tile
        processed = preprocess_query_frame(
            meta_tile,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        ref_mask = self.model.predict(processed)

        confidence = compute_histogram_confidence(
            query_semantic_map, ref_mask
        )

        return {
            "confidence": confidence,
            "matched_pairs": None,    # legacy field; no longer used
            "total_query_centroids": None,
            "match_ratio": confidence,
        }

