"""
SuperPoint + LightGlue feature matcher wrapper.
"""

import torch
import numpy as np
from typing import Dict, Optional


def _to_tensor_gray(image: np.ndarray, device: str) -> torch.Tensor:
    """Convert HWC uint8 image to (1, 1, H, W) float tensor for SuperPoint."""
    if image.ndim == 3 and image.shape[2] == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    elif image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = image[..., 0].astype(np.float32)
    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


class SuperPointLightGlueMatcher:
    """
    Feature matcher using SuperPoint (detector/descriptor) + LightGlue (matcher).
    Weights are auto-downloaded on first use.
    """

    def __init__(self, max_num_keypoints: int = 2048, device: str = "cuda"):
        from lightglue import SuperPoint, LightGlue

        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    @torch.no_grad()
    def match(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        Match two images.

        Args:
            img1, img2: HWC uint8 numpy arrays (RGB or grayscale).

        Returns:
            dict with keys:
                keypoints1  (N, 2) float array
                keypoints2  (M, 2) float array
                matches     (K, 2) int array — index pairs
                match_scores (K,) float array
                num_matches  int
        """
        feats0 = self.extract_features(img1)
        feats1 = self.extractor.extract(_to_tensor_gray(img2, self.device))
        return self._run_matcher(feats0, feats1)

    @torch.no_grad()
    def extract_features(self, img: np.ndarray) -> Dict:
        """Extract SuperPoint features from a single image.

        Call once per query frame and reuse with match_precomputed()
        to avoid redundant extraction across many tile matches.
        """
        return self.extractor.extract(_to_tensor_gray(img, self.device))

    @torch.no_grad()
    def match_precomputed(self, feats0: Dict, img2: np.ndarray) -> Dict:
        """Match pre-extracted query features against a new image.

        Args:
            feats0: Pre-computed features from extract_features().
            img2:   Reference tile as HWC uint8 array.

        Returns same dict format as match().
        """
        feats1 = self.extractor.extract(_to_tensor_gray(img2, self.device))
        return self._run_matcher(feats0, feats1)

    @torch.no_grad()
    def match_both_precomputed(self, feats0: Dict, feats1: Dict) -> Dict:
        """Match two pre-extracted feature dicts (both query and reference).

        Use when reference features have been loaded from a precomputed
        feature store instead of extracted at runtime.

        Args:
            feats0: Pre-computed query features from extract_features().
            feats1: Pre-computed reference features (from feature store or
                    extract_features()).

        Returns same dict format as match().
        """
        return self._run_matcher(feats0, feats1)

    def _run_matcher(self, feats0: Dict, feats1: Dict) -> Dict:
        """Run LightGlue on two pre-extracted feature dicts."""
        matches_out = self.matcher({"image0": feats0, "image1": feats1})

        kpts1 = feats0["keypoints"][0].cpu().numpy()
        kpts2 = feats1["keypoints"][0].cpu().numpy()
        matches0 = matches_out["matches0"][0].cpu().numpy()
        scores = matches_out["matching_scores0"][0].cpu().numpy()

        valid = matches0 >= 0
        match_pairs = np.column_stack([np.where(valid)[0], matches0[valid].astype(int)])
        match_scores = scores[valid]

        return {
            "keypoints1": kpts1,
            "keypoints2": kpts2,
            "matches": match_pairs,
            "match_scores": match_scores,
            "num_matches": len(match_pairs),
        }


def initialize_matcher(device: str = "cuda",
                       max_num_keypoints: int = 2048) -> SuperPointLightGlueMatcher:
    """Factory function expected by other modules."""
    return SuperPointLightGlueMatcher(max_num_keypoints=max_num_keypoints, device=device)
