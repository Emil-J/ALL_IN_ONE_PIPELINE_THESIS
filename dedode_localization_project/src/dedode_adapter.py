"""
DeDoDe v2 Adapter

Wrapper for DeDoDe (Detect, Describe, and Match) feature matching.
Isolates DeDoDe API calls for easy version adaptation.

DeDoDe is used as the VISUAL CORRECTION module for local-area matching
against IMU-constrained candidate tiles.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Tuple, Dict, Optional
import warnings


class DeDoDeMatcher:
    """
    DeDoDe feature detector and matcher
    
    Provides a stable interface for DeDoDe regardless of specific version.
    Supports both kornia.feature.DeDoDe and standalone DeDoDe implementations.
    """
    
    def __init__(self,
                 detector_weights: str = "L-upright",
                 descriptor_weights: str = "B-upright",
                 num_keypoints: int = 5000,
                 device: str = "cuda",
                 use_kornia: bool = True):
        """
        Initialize DeDoDe matcher
        
        Args:
            detector_weights: Detector model ("L-upright", "L-C4", etc.)
            descriptor_weights: Descriptor model ("B-upright", "B-C4", etc.)
            num_keypoints: Maximum number of keypoints to detect
            device: "cuda" or "cpu"
            use_kornia: If True, use kornia.feature.DeDoDe (recommended)
        """
        self.detector_weights = detector_weights
        self.descriptor_weights = descriptor_weights
        self.num_keypoints = num_keypoints
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_kornia = use_kornia
        
        # Load models
        self.detector = None
        self.descriptor = None
        self.matcher = None
        
        self._load_models()
    
    def _load_models(self):
        """Load DeDoDe detector, descriptor, and matcher"""
        if self.use_kornia:
            self._load_kornia_dedode()
        else:
            # Fallback: try kornia anyway since standalone DeDoDe pip package may not exist
            self._load_kornia_dedode()
    
    def _load_kornia_dedode(self):
        """Load DeDoDe from kornia.feature (kornia >= 0.8)"""
        try:
            from kornia.feature import DeDoDe, LightGlueMatcher
            
            # Use DeDoDe.from_pretrained() - the correct API for kornia 0.8+
            self.dedode_model = DeDoDe.from_pretrained(
                detector_weights=self.detector_weights,
                descriptor_weights=self.descriptor_weights
            ).to(self.device).eval()
            
            # LightGlueMatcher requires 'dedodeb' for B-descriptor, 'dedodeg' for G-descriptor
            matcher_key = 'dedodeb' if 'B' in self.descriptor_weights else 'dedodeg'
            self.matcher = LightGlueMatcher(matcher_key).to(self.device).eval()
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import kornia.feature.DeDoDe. "
                f"Install kornia with: pip install kornia kornia-rs\nError: {e}"
            )
    
    def detect_and_describe(self, 
                           image: Union[np.ndarray, torch.Tensor],
                           resize: Optional[int] = None) -> Dict:
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input image (H, W, 3) as numpy or (C, H, W) as tensor
            resize: If provided, resize image to this size (single value = square)
        
        Returns:
            Dict with keys:
                - keypoints: (N, 2) array of keypoint coordinates
                - descriptors: (N, D) array of descriptors
                - confidence: (N,) array of detection confidence scores
                - image_tensor: Preprocessed image tensor (for reference)
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image_tensor = self._numpy_to_tensor(image)
        else:
            image_tensor = image
        
        # Resize if needed
        original_size = image_tensor.shape[-2:]
        if resize:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(resize, resize) if isinstance(resize, int) else resize,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            # kornia DeDoDe.from_pretrained() returns (keypoints, scores, descriptors)
            # Input must be (B, C, H, W)
            batch_input = image_tensor.unsqueeze(0)
            keypoints, confidence, descriptors = self.dedode_model(batch_input)
            
            # Remove batch dimension: (1, N, 2) -> (N, 2), etc.
            keypoints = keypoints[0].cpu().numpy()    # (N, 2)
            descriptors = descriptors[0].cpu().numpy() # (N, D)
            confidence = confidence[0].cpu().numpy()   # (N,)
        
        # Scale keypoints back to original size if resized
        if resize and original_size != image_tensor.shape[-2:]:
            scale_y = original_size[0] / image_tensor.shape[-2]
            scale_x = original_size[1] / image_tensor.shape[-1]
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y
        
        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "confidence": confidence,
            "image_tensor": image_tensor
        }
    
    def match(self,
              desc1: Dict,
              desc2: Dict,
              threshold: float = 0.2) -> Dict:
        """
        Match descriptors between two images
        
        Args:
            desc1, desc2: Descriptor dicts from detect_and_describe()
            threshold: Matching threshold
        
        Returns:
            Dict with keys:
                - matches: (M, 2) array of match indices
                - match_confidence: (M,) array of match confidences
        """
        kp1 = torch.from_numpy(desc1["keypoints"]).to(self.device)
        kp2 = torch.from_numpy(desc2["keypoints"]).to(self.device)
        desc1_t = torch.from_numpy(desc1["descriptors"]).to(self.device)
        desc2_t = torch.from_numpy(desc2["descriptors"]).to(self.device)
        
        with torch.no_grad():
            # kornia LightGlueMatcher returns (dists, idxs)
            # dists shape: (M, 1), idxs shape: (M, 2)
            dists, idxs = self.matcher(
                desc1_t.unsqueeze(0),
                desc2_t.unsqueeze(0),
                kp1.unsqueeze(0),
                kp2.unsqueeze(0)
            )
            matches = idxs.cpu().numpy()  # (M, 2)
            confidence = dists.cpu().numpy().ravel() if len(dists) > 0 else np.array([])
        
        return {
            "matches": matches,
            "match_confidence": confidence
        }
    
    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = image[np.newaxis, :, :]  # Add channel dim
        
        return torch.from_numpy(image)


def compute_homography(kp1: np.ndarray, 
                      kp2: np.ndarray,
                      matches: np.ndarray,
                      ransac_threshold: float = 4.0,
                      min_matches: int = 8) -> Tuple[Optional[np.ndarray], np.ndarray, Dict]:
    """
    Compute homography using RANSAC
    
    Args:
        kp1, kp2: Keypoints from both images (N1, 2) and (N2, 2)
        matches: Match indices (M, 2)
        ransac_threshold: RANSAC inlier threshold in pixels
        min_matches: Minimum matches required
    
    Returns:
        - homography: 3x3 homography matrix or None if failed
        - inlier_mask: Boolean mask of inliers
        - stats: Dict with num_inliers, inlier_ratio, reprojection_error
    """
    if len(matches) < min_matches:
        return None, np.array([]), {
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_reproj_error": np.inf,
            "median_reproj_error": np.inf
        }
    
    # Extract matched keypoints
    pts1 = kp1[matches[:, 0]]
    pts2 = kp2[matches[:, 1]]
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(
        pts1, pts2,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995
    )
    
    if H is None or mask is None:
        return None, np.array([]), {
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_reproj_error": np.inf,
            "median_reproj_error": np.inf
        }
    
    inlier_mask = mask.ravel().astype(bool)
    num_inliers = inlier_mask.sum()
    inlier_ratio = num_inliers / len(matches)
    
    # Compute reprojection error for inliers
    if num_inliers > 0:
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]
        
        # Project pts1 using homography
        pts1_homo = np.hstack([pts1_inliers, np.ones((num_inliers, 1))])
        pts1_projected = (H @ pts1_homo.T).T
        pts1_projected = pts1_projected[:, :2] / pts1_projected[:, 2:3]
        
        # Compute errors
        errors = np.linalg.norm(pts1_projected - pts2_inliers, axis=1)
        mean_error = np.mean(errors)
        median_error = np.median(errors)
    else:
        mean_error = np.inf
        median_error = np.inf
    
    stats = {
        "num_inliers": int(num_inliers),
        "inlier_ratio": float(inlier_ratio),
        "mean_reproj_error": float(mean_error),
        "median_reproj_error": float(median_error)
    }
    
    return H, inlier_mask, stats


def estimate_position_from_homography(H: np.ndarray,
                                     query_size: Tuple[int, int],
                                     reference_lat: float,
                                     reference_lon: float,
                                     tile_size_meters: Tuple[float, float]) -> Tuple[float, float]:
    """
    Estimate position offset from homography
    
    This is a simplified version that assumes the homography primarily
    encodes translation. For more accurate refinement, a full pose estimation
    would be needed.
    
    Args:
        H: 3x3 homography matrix
        query_size: (height, width) of query image
        reference_lat, reference_lon: Center coordinates of reference tile
        tile_size_meters: (height, width) of tile in meters
    
    Returns:
        (estimated_lat, estimated_lon) - refined position
    """
    # Extract translation from homography
    # H maps query points to reference points
    # We want the center offset
    
    # Transform query center to reference frame
    query_center = np.array([[query_size[1]/2, query_size[0]/2, 1]]).T
    ref_center = H @ query_center
    ref_center = ref_center[:2] / ref_center[2]
    
    # Compute pixel offset from reference center
    ref_img_center = np.array([query_size[1]/2, query_size[0]/2])
    pixel_offset = ref_center.ravel() - ref_img_center
    
    # Convert pixel offset to meters (approximate)
    meters_per_pixel_x = tile_size_meters[1] / query_size[1]
    meters_per_pixel_y = tile_size_meters[0] / query_size[0]
    
    offset_east = pixel_offset[0] * meters_per_pixel_x
    offset_north = -pixel_offset[1] * meters_per_pixel_y  # Y is down in images
    
    # Convert to lat/lon offset
    from .tms_utils import ned_to_latlon
    
    est_lat, est_lon, _ = ned_to_latlon(
        offset_north, offset_east, 0,
        reference_lat, reference_lon, 0
    )
    
    return est_lat, est_lon
