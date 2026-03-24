"""
Image processing utilities
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import torch


def load_image(filepath: Union[str, Path], 
               as_rgb: bool = True,
               as_numpy: bool = True) -> np.ndarray:
    """
    Load image from file
    
    Args:
        filepath: Path to image
        as_rgb: If True, convert to RGB (default True)
        as_numpy: If True, return numpy array, else PIL Image
    
    Returns:
        Image as numpy array (H, W, 3) or PIL Image
    """
    img = Image.open(filepath)
    
    if as_rgb and img.mode != 'RGB':
        img = img.convert('RGB')
    
    if as_numpy:
        return np.array(img)
    return img


def save_image(img: Union[np.ndarray, Image.Image],
               filepath: Union[str, Path],
               quality: int = 95):
    """Save image to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img.save(filepath, quality=quality)


def resize_image(img: np.ndarray, 
                 size: Union[int, Tuple[int, int]],
                 interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image
    
    Args:
        img: Input image (H, W, C)
        size: Target size - if int, resize to (size, size), else (width, height)
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized image
    """
    if isinstance(size, int):
        size = (size, size)
    
    return cv2.resize(img, size, interpolation=interpolation)


def normalize_image(img: np.ndarray,
                    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image using ImageNet statistics
    
    Args:
        img: Image as numpy array (H, W, 3), values in [0, 255] or [0, 1]
        mean: Mean values per channel
        std: Std values per channel
    
    Returns:
        Normalized image
    """
    img = img.astype(np.float32)
    
    # Convert to [0, 1] if needed
    if img.max() > 1.0:
        img /= 255.0
    
    # Normalize
    img = (img - np.array(mean)) / np.array(std)
    
    return img


def to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor
    
    Args:
        img: Image (H, W, C) in [0, 255] or [0, 1]
        normalize: If True, apply ImageNet normalization
    
    Returns:
        Tensor (C, H, W)
    """
    img = img.astype(np.float32)
    
    if img.max() > 1.0:
        img /= 255.0
    
    if normalize:
        img = normalize_image(img)
    
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    
    return torch.from_numpy(img)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor back to numpy image
    
    Args:
        tensor: Tensor (C, H, W) or (B, C, H, W)
    
    Returns:
        Image (H, W, C) in [0, 255]
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()
    
    # CHW -> HWC
    img = np.transpose(img, (1, 2, 0))
    
    # Denormalize if needed
    if img.min() < -1.0 or img.max() > 2.0:
        # Likely normalized, denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
    
    # Clip and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def crop_image(img: np.ndarray,
               x: int, y: int,
               width: int, height: int) -> np.ndarray:
    """Crop image to specified rectangle"""
    return img[y:y+height, x:x+width]


def pad_image(img: np.ndarray,
              target_size: Tuple[int, int],
              fill_value: int = 0) -> np.ndarray:
    """
    Pad image to target size
    
    Args:
        img: Input image
        target_size: (height, width)
        fill_value: Value to use for padding
    
    Returns:
        Padded image
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    if h >= target_h and w >= target_w:
        return img
    
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # Pad evenly on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if img.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    
    return np.pad(img, padding, mode='constant', constant_values=fill_value)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Useful for improving contrast in low-contrast images
    
    Args:
        img: Input image (grayscale or RGB)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    if img.ndim == 3:
        # Convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img)


def compute_image_similarity(img1: np.ndarray, img2: np.ndarray, method: str = 'ssim') -> float:
    """
    Compute similarity between two images
    
    Args:
        img1, img2: Images to compare
        method: 'ssim' or 'mse'
    
    Returns:
        Similarity score
    """
    if method == 'mse':
        return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))
    elif method == 'ssim':
        from skimage.metrics import structural_similarity as ssim
        if img1.ndim == 3:
            return ssim(img1, img2, multichannel=True, channel_axis=2)
        else:
            return ssim(img1, img2)
    else:
        raise ValueError(f"Unknown method: {method}")


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by 90, 180, or 270 degrees
    
    Args:
        img: Input image
        angle: Rotation angle (0, 90, 180, 270)
    
    Returns:
        Rotated image
    """
    if angle == 0:
        return img
    elif angle == 90:
        return np.rot90(img, k=1)
    elif angle == 180:
        return np.rot90(img, k=2)
    elif angle == 270:
        return np.rot90(img, k=3)
    else:
        raise ValueError(f"Angle must be 0, 90, 180, or 270, got {angle}")


def convert_mask_to_rgb(class_mask: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Convert class mask to RGB visualization
    
    Args:
        class_mask: (H, W) array of class indices
        color_map: Dict mapping class_id -> (R, G, B)
    
    Returns:
        RGB visualization (H, W, 3)
    """
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        rgb_mask[class_mask == class_id] = color
    
    return rgb_mask


def rgb_to_class_mask(rgb_mask: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Convert RGB mask back to class indices
    
    Args:
        rgb_mask: (H, W, 3) RGB visualization
        color_map: Dict mapping class_id -> (R, G, B)
    
    Returns:
        Class mask (H, W) with class indices
    """
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        mask = np.all(rgb_mask == color, axis=-1)
        class_mask[mask] = class_id
    
    return class_mask
