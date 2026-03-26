"""
Image loading, resizing, and preprocessing utilities.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


def load_image(filepath: Union[str, Path], as_rgb: bool = True) -> np.ndarray:
    """Load image from disk as numpy array (H, W, 3) RGB."""
    img = Image.open(filepath)
    if as_rgb:
        img = img.convert("RGB")
    return np.array(img)


def preprocess_query_frame(image: np.ndarray,
                           resize_w: int = 512,
                           resize_h: int = 288,
                           target_size: int = 512) -> np.ndarray:
    """
    Resize query frame (1920x1079) to 512x288 then pad to 512x512.
    Used ONLY for the semantic model which requires 512x512 input.
    Returns (target_size, target_size, 3) uint8 array.
    """
    pil = Image.fromarray(image)
    pil = pil.resize((resize_w, resize_h), Image.LANCZOS)
    resized = np.array(pil)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - resize_h) // 2
    canvas[pad_top:pad_top + resize_h, :resize_w] = resized
    return canvas


def resize_for_matching(image: np.ndarray,
                        max_size: int = 800) -> np.ndarray:
    """
    Center-crop then resize for feature matching (no padding).
    Crops a square region from the center of the image first,
    then resizes to max_size if needed. Preserves aspect ratio
    and detail from the most relevant part of the frame.
    Returns (H', W', 3) uint8 array.
    """
    h, w = image.shape[:2]
    # Center-crop to a square using the shorter dimension
    crop_size = min(h, w)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    cropped = image[y0:y0 + crop_size, x0:x0 + crop_size]

    if crop_size <= max_size:
        return cropped
    pil = Image.fromarray(cropped)
    pil = pil.resize((max_size, max_size), Image.LANCZOS)
    return np.array(pil)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB (H, W, 3) to grayscale (H, W)."""
    if image.ndim == 2:
        return image
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
