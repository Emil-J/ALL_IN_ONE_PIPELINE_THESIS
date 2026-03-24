"""
Semantic segmentation model — UNet++ EfficientNet-B3 + scSE.
Loads the trained checkpoint and provides per-frame inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from PIL import Image


class SemanticModel:
    """
    Wrapper around the UNet++ segmentation model.
    Input: 512×512 RGB.  Output: (H, W) class mask with indices 0-5.
    """

    CLASS_NAMES = {
        0: "waterbodies",
        1: "forest_trees",
        2: "land",
        3: "railway",
        4: "roads",
        5: "buildings",
    }

    COLOR_MAP = {
        0: (4, 4, 255),
        1: (0, 167, 2),
        2: (243, 255, 150),
        3: (193, 105, 53),
        4: (255, 0, 231),
        5: (150, 150, 150),
    }

    def __init__(self, model_path: Union[str, Path], device: str = "cuda",
                 input_size: int = 512, num_classes: int = 6):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = self._load_model(Path(model_path))
        self.model.eval()

    # ── Loading ──────────────────────────────────────────────────

    def _load_model(self, model_path: Path):
        import segmentation_models_pytorch as smp

        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
            decoder_attention_type="scse",
            activation=None,
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device)

    # ── Inference ────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a single image.

        Args:
            image: (H, W, 3) uint8 RGB array. Should already be 512×512
                   (with padding applied if needed).

        Returns:
            (H, W) uint8 class mask (values 0-5).
        """
        tensor = self._preprocess(image)
        logits = self.model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Normalise and convert HWC uint8 → NCHW float tensor."""
        img = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    # ── Visualisation helpers ────────────────────────────────────

    def mask_to_rgb(self, mask: np.ndarray,
                    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
        """Convert class mask → RGB visualisation."""
        cmap = color_map or self.COLOR_MAP
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, colour in cmap.items():
            rgb[mask == cid] = colour
        return rgb


def load_semantic_model(model_path: Union[str, Path],
                        device: str = "cuda") -> SemanticModel:
    """Factory function expected by other modules."""
    return SemanticModel(model_path, device=device)
