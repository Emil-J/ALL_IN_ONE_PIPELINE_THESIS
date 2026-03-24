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
    Resize query frame (1920×1079) to 512×288 then pad to 512×512.
    Returns (target_size, target_size, 3) uint8 array.
    """
    pil = Image.fromarray(image)
    pil = pil.resize((resize_w, resize_h), Image.LANCZOS)
    resized = np.array(pil)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - resize_h) // 2
    canvas[pad_top:pad_top + resize_h, :resize_w] = resized
    return canvas


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB (H, W, 3) to grayscale (H, W)."""
    if image.ndim == 2:
        return image
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
