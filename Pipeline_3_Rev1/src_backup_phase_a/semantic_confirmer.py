"""
Module 11 — Semantic Confirmer.

Centroid-based double-confirmation between query semantic map (Branch A)
and meta-tile semantic map (MDPI method, doi:10.3390/rs17101671).
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import ndimage

from src.image_utils import preprocess_query_frame


class SemanticConfirmer:
    """
    Semantic double-confirmation:
      1. segment()       — run UNet++ on an image → class mask
      2. extract_centroids() — connected-component centroids per class
      3. match_centroids()   — nearest-same-class pairing
      4. confirm()       — orchestrate steps 1-3 between query & meta-tile
    """

    def __init__(self, semantic_model, config):
        """
        Args:
            semantic_model: SemanticModel instance (src.semantic_model).
            config: config module.
        """
        self.model = semantic_model
        self.cfg = config

    # ─── 11.2  Segment ───────────────────────────────────────────

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess and run semantic segmentation.

        Args:
            image: Raw frame (any size).  Will be resized + padded to 512×512.

        Returns:
            (512, 512) uint8 class mask (values 0-5).
        """
        processed = preprocess_query_frame(
            image,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        return self.model.predict(processed)

    # ─── 11.3  Extract centroids ─────────────────────────────────

    @staticmethod
    def extract_centroids(semantic_mask: np.ndarray) -> List[Dict]:
        """
        Extract per-class connected-component centroids.

        Returns:
            list of {'class': int, 'cx': float, 'cy': float, 'area': int}
        """
        centroids = []
        for cls_id in np.unique(semantic_mask):
            binary = (semantic_mask == cls_id).astype(np.uint8)
            labelled, n_components = ndimage.label(binary)
            for comp_id in range(1, n_components + 1):
                ys, xs = np.where(labelled == comp_id)
                area = len(xs)
                if area < 10:          # skip tiny fragments
                    continue
                cx = float(xs.mean())
                cy = float(ys.mean())
                centroids.append({
                    "class": int(cls_id),
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                })
        return centroids

    # ─── 11.4  Match centroids ───────────────────────────────────

    def match_centroids(self, query_centroids: List[Dict],
                        reference_centroids: List[Dict]) -> Dict:
        """
        For each query centroid, find nearest reference centroid of the same
        class within CENTROID_MATCH_DISTANCE_THRESHOLD_PX.

        Returns:
            dict with matched_pairs, total_query_centroids,
            match_ratio, confidence.
        """
        threshold_px = self.cfg.CENTROID_MATCH_DISTANCE_THRESHOLD_PX
        min_pairs = self.cfg.SEMANTIC_CONFIRM_MIN_PAIRS

        # Build class-indexed lookup for reference centroids
        ref_by_class: Dict[int, List[Dict]] = {}
        for c in reference_centroids:
            ref_by_class.setdefault(c["class"], []).append(c)

        matched = 0
        matched_area_sum = 0
        for qc in query_centroids:
            cls = qc["class"]
            refs = ref_by_class.get(cls, [])
            best_dist = float("inf")
            best_ref = None
            for rc in refs:
                d = np.hypot(qc["cx"] - rc["cx"], qc["cy"] - rc["cy"])
                if d < best_dist:
                    best_dist = d
                    best_ref = rc
            if best_dist <= threshold_px and best_ref is not None:
                matched += 1
                matched_area_sum += qc["area"]

        total = max(len(query_centroids), 1)
        ratio = matched / total

        if matched >= min_pairs and total > 0:
            mean_area = matched_area_sum / max(matched, 1)
            # Normalise area contribution (cap at 1.0)
            area_factor = min(mean_area / 5000.0, 1.0)
            confidence = ratio * (0.5 + 0.5 * area_factor)
        else:
            confidence = 0.0

        return {
            "matched_pairs": matched,
            "total_query_centroids": len(query_centroids),
            "match_ratio": ratio,
            "confidence": confidence,
        }

    # ─── 11.5  Confirm ───────────────────────────────────────────

    def confirm(self, query_semantic_map: np.ndarray,
                meta_tile: np.ndarray) -> Dict:
        """
        Semantic double-confirmation.

        Args:
            query_semantic_map: Pre-computed query class mask (from Branch A).
                                Do NOT re-run inference on the query.
            meta_tile: RGB meta-tile image.

        Returns:
            dict with matched_pairs, match_ratio, confidence,
            query_centroids, meta_tile_centroids.
        """
        # Segment meta-tile (no pad — already sized by build_meta_tile)
        meta_semantic = self.model.predict(
            preprocess_query_frame(
                meta_tile,
                resize_w=self.cfg.QUERY_RESIZE_WIDTH,
                resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
                target_size=self.cfg.SEMANTIC_INPUT_SIZE,
            )
        )

        q_centroids = self.extract_centroids(query_semantic_map)
        m_centroids = self.extract_centroids(meta_semantic)

        result = self.match_centroids(q_centroids, m_centroids)
        result["query_centroids"] = len(q_centroids)
        result["meta_tile_centroids"] = len(m_centroids)
        return result
