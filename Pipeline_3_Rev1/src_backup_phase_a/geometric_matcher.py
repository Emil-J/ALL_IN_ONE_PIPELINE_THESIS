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
        tensor1 = _to_tensor_gray(img1, self.device)
        tensor2 = _to_tensor_gray(img2, self.device)

        feats0 = self.extractor.extract(tensor1)
        feats1 = self.extractor.extract(tensor2)

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
