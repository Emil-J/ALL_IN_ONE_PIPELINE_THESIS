"""
Semantic Segmentation Adapter

Loads semantic segmentation model and provides per-frame inference with caching.
Reuses existing model architecture and weights from SemanticTerrainSegmentationModel.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hashlib


class SemanticSegmentationModel:
    """
    Semantic segmentation model wrapper
    
    Loads trained UNet++ model and provides per-frame inference with optional caching.
    Reuses existing model from SemanticTerrainSegmentationModel/best.pth
    """
    
    def __init__(self,
                 model_path: Path,
                 input_size: int = 256,
                 num_classes: int = 6,
                 device: str = "cuda",
                 cache_dir: Optional[Path] = None):
        """
        Initialize semantic segmentation model
        
        Args:
            model_path: Path to model checkpoint (.pth)
            input_size: Input image size (256 or 512)
            num_classes: Number of segmentation classes
            device: "cuda" or "cpu"
            cache_dir: Directory for caching predictions (optional)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _load_model(self):
        """Load model architecture and weights"""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch not installed. "
                "Install with: pip install segmentation-models-pytorch"
            )
        
        # Create model architecture (must match training config)
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,  # Will load from checkpoint
            in_channels=3,
            classes=self.num_classes,
            decoder_attention_type='scse',  # Spatial and channel squeeze & excitation
            activation=None  # Apply softmax separately
        )
        
        # Load checkpoint
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        return model
    
    def predict(self, 
                image: Union[np.ndarray, Path, str],
                return_rgb: bool = False,
                color_map: Optional[Dict] = None,
                use_cache: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run semantic segmentation on image
        
        Args:
            image: Input image as numpy array or file path
            return_rgb: If True, also return RGB visualization
            color_map: Dict mapping class_id -> (R, G, B) for visualization
            use_cache: Use cached prediction if available
        
        Returns:
            class_mask (H, W) or (class_mask, rgb_mask) if return_rgb=True
        """
        # Load image if path provided
        if isinstance(image, (Path, str)):
            image_path = Path(image)
            
            # Check cache
            if use_cache and self.cache_dir:
                cached_mask = self._load_from_cache(image_path)
                if cached_mask is not None:
                    if return_rgb:
                        rgb_mask = self._class_mask_to_rgb(cached_mask, color_map)
                        return cached_mask, rgb_mask
                    return cached_mask
            
            image = np.array(Image.open(image_path).convert('RGB'))
        else:
            image_path = None
        
        original_size = image.shape[:2]
        
        # Preprocess
        transformed = self.transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (original_size[1], original_size[0]),
                Image.NEAREST
            )
        )
        
        # Save to cache
        if use_cache and self.cache_dir and image_path:
            self._save_to_cache(image_path, pred_resized)
        
        # Optional RGB visualization
        if return_rgb:
            rgb_mask = self._class_mask_to_rgb(pred_resized, color_map)
            return pred_resized, rgb_mask
        
        return pred_resized
    
    def _class_mask_to_rgb(self, class_mask: np.ndarray, color_map: Optional[Dict]) -> np.ndarray:
        """Convert class mask to RGB visualization"""
        if color_map is None:
            # Default color map
            color_map = {
                0: (4, 4, 255),
                1: (0, 167, 2),
                2: (243, 255, 150),
                3: (193, 105, 53),
                4: (255, 0, 231),
                5: (150, 150, 150)
            }
        
        h, w = class_mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in color_map.items():
            rgb_mask[class_mask == class_id] = color
        
        return rgb_mask
    
    def _get_cache_path(self, image_path: Path) -> Path:
        """Generate cache file path for image"""
        # Use hash of full path to avoid collisions
        path_hash = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
        filename = f"{image_path.stem}_{path_hash}.npy"
        return self.cache_dir / filename
    
    def _save_to_cache(self, image_path: Path, mask: np.ndarray):
        """Save prediction to cache"""
        cache_path = self._get_cache_path(image_path)
        np.save(cache_path, mask)
    
    def _load_from_cache(self, image_path: Path) -> Optional[np.ndarray]:
        """Load prediction from cache"""
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def clear_cache(self):
        """Clear all cached predictions"""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()


def extract_landmarks(class_mask: np.ndarray,
                     filter_classes: list = None,
                     min_area: int = 100) -> list:
    """
    Extract landmark centroids from semantic mask
    
    Reuses logic from existing localization notebook.
    
    Args:
        class_mask: Segmentation mask (H, W) with class indices
        filter_classes: List of class IDs to extract (e.g., [2, 4, 5] for land, roads, buildings)
        min_area: Minimum pixel area for a landmark
    
    Returns:
        List of landmark dicts with keys: centroid, area, class_id, bbox
    """
    import cv2
    
    landmarks = []
    
    # Get classes to process
    if filter_classes is None:
        filter_classes = list(np.unique(class_mask))
    
    for class_id in filter_classes:
        # Extract binary mask for this class
        binary_mask = (class_mask == class_id).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < min_area:
                continue
            
            centroid = centroids[i]
            bbox = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT]
            )
            
            landmarks.append({
                'centroid': tuple(centroid),
                'area': int(area),
                'class_id': int(class_id),
                'bbox': bbox
            })
    
    return landmarks


def compute_landmark_features(landmarks: list, 
                              max_triplets: int = 50,
                              min_distance: float = 10.0) -> np.ndarray:
    """
    Compute geometric feature vector from landmarks
    
    Builds triplets of landmarks and computes geometric invariants.
    Reuses logic from existing localization notebook.
    
    Args:
        landmarks: List of landmark dicts
        max_triplets: Maximum number of triplets to compute
        min_distance: Minimum distance between landmarks in a triplet
    
    Returns:
        Feature vector (flattened array of triplet features)
    """
    if len(landmarks) < 3:
        return np.array([])
    
    # Extract centroids
    centroids = np.array([lm['centroid'] for lm in landmarks])
    
    # Build triplets
    triplets = []
    for i in range(len(landmarks)):
        for j in range(i+1, len(landmarks)):
            for k in range(j+1, len(landmarks)):
                # Check minimum distance
                p1, p2, p3 = centroids[i], centroids[j], centroids[k]
                
                d12 = np.linalg.norm(p1 - p2)
                d23 = np.linalg.norm(p2 - p3)
                d13 = np.linalg.norm(p1 - p3)
                
                if min(d12, d23, d13) < min_distance:
                    continue
                
                # Compute geometric features (scale-invariant)
                # Normalize distances by largest
                max_dist = max(d12, d23, d13)
                feat = np.array([
                    d12 / max_dist,
                    d23 / max_dist,
                    d13 / max_dist
                ])
                
                # Add class info
                class_feat = np.array([
                    landmarks[i]['class_id'],
                    landmarks[j]['class_id'],
                    landmarks[k]['class_id']
                ])
                
                triplets.append(np.concatenate([feat, class_feat]))
                
                if len(triplets) >= max_triplets:
                    break
            if len(triplets) >= max_triplets:
                break
        if len(triplets) >= max_triplets:
            break
    
    if not triplets:
        return np.array([])
    
    # Flatten and return
    return np.concatenate(triplets)
