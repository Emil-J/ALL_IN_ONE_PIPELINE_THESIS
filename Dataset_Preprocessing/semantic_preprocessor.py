"""
Semantic Reference Tile Preprocessor.

Runs UNet++ EfficientNet-B3 semantic segmentation on all reference aerial
tiles and saves RGB-encoded prediction masks in the same TMS directory
structure.  This is the offline counterpart of the runtime semantic model.

Refactored from Inference_Reference_Map.ipynb.

Incremental: tiles whose prediction already exists on disk are skipped
unless --force is passed.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Default color map matching pipeline config
DEFAULT_COLOR_MAP = {
    0: (4, 4, 255),       # waterbodies
    1: (0, 167, 2),       # forest_trees
    2: (243, 255, 150),   # land
    3: (193, 105, 53),    # railway
    4: (255, 0, 231),     # roads
    5: (150, 150, 150),   # buildings
}

DEFAULT_CLASS_NAMES = [
    "waterbodies", "forest_trees", "land", "railway", "roads", "buildings"
]


def _load_semantic_model(model_path: Path, device: str, num_classes: int = 6):
    """Load UNet++ EfficientNet-B3 model from checkpoint."""
    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,  # loaded from checkpoint
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse",
        activation=None,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        iou = checkpoint.get("best_iou", "?")
        logger.info("Loaded semantic model: epoch=%s, best_iou=%s", epoch, iou)
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded semantic model (legacy checkpoint)")

    model.eval()
    return model


def _preprocess_tile(image: np.ndarray, device: str) -> torch.Tensor:
    """Normalise and convert HWC uint8 → NCHW float tensor.

    Uses ImageNet normalization (same as training and runtime).
    """
    img = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _decode_segmap(mask: np.ndarray, color_map: Dict) -> np.ndarray:
    """Convert class mask (H,W) → RGB image (H,W,3) using color map."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_map.items():
        out[mask == k] = v
    return out


def _discover_tiles(aerial_dir: Path, zoom: int = 16):
    """Yield (tile_x, tile_y, tile_path) for all tiles in TMS structure."""
    zoom_dir = aerial_dir / str(zoom)
    if not zoom_dir.exists():
        raise FileNotFoundError(f"Zoom directory not found: {zoom_dir}")

    for x_dir in sorted(zoom_dir.iterdir()):
        if not x_dir.is_dir():
            continue
        try:
            tile_x = int(x_dir.name)
        except ValueError:
            continue
        for tile_file in sorted(x_dir.iterdir()):
            if tile_file.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            try:
                tile_y = int(tile_file.stem)
            except ValueError:
                continue
            yield tile_x, tile_y, tile_file


def run_semantic_preprocessing(
    aerial_dir: Path,
    prediction_dir: Path,
    model_path: Path,
    *,
    device: str = "cuda",
    zoom: int = 16,
    num_classes: int = 6,
    color_map: Optional[Dict] = None,
    force: bool = False,
) -> dict:
    """
    Run semantic segmentation on all reference tiles and save predictions.

    Args:
        aerial_dir:     Path to TMS aerial tile directory.
        prediction_dir: Path to output prediction tile directory.
        model_path:     Path to semantic model checkpoint.
        device:         PyTorch device.
        zoom:           TMS zoom level.
        num_classes:    Number of semantic classes.
        color_map:      Class ID → RGB color mapping. Defaults to pipeline config.
        force:          If True, overwrite existing prediction tiles.

    Returns:
        dict with processing stats.
    """
    cmap = color_map or DEFAULT_COLOR_MAP

    # Auto-detect device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"

    # Discover tiles
    tiles = list(_discover_tiles(aerial_dir, zoom))
    total = len(tiles)
    logger.info("Discovered %d reference tiles in %s", total, aerial_dir)

    if total == 0:
        raise ValueError(f"No tiles found in {aerial_dir}/{zoom}/")

    # Load model
    model = _load_semantic_model(model_path, device, num_classes)

    stats = {"tiles_processed": 0, "tiles_skipped": 0, "tiles_failed": 0,
             "total_tiles": total, "class_pixel_counts": {c: 0 for c in range(num_classes)}}

    t0 = time.time()

    for i, (tx, ty, tile_path) in enumerate(tiles):
        # Output path preserves TMS structure
        rel_path = Path(str(zoom)) / str(tx) / f"{ty}.png"
        pred_path = prediction_dir / rel_path

        # Skip if exists (incremental)
        if not force and pred_path.exists():
            stats["tiles_skipped"] += 1
            continue

        try:
            # Load and preprocess
            img = np.array(Image.open(tile_path).convert("RGB"))
            tensor = _preprocess_tile(img, device)

            # Inference
            with torch.no_grad():
                logits = model(tensor)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Accumulate class stats
            for c in range(num_classes):
                stats["class_pixel_counts"][c] += int((pred == c).sum())

            # Encode to RGB and save
            pred_rgb = _decode_segmap(pred, cmap)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_rgb).save(pred_path)

            stats["tiles_processed"] += 1

        except Exception as e:
            logger.error("Failed to process tile (%d, %d) at %s: %s",
                         tx, ty, tile_path, e)
            stats["tiles_failed"] += 1

        # Progress
        done = stats["tiles_processed"] + stats["tiles_skipped"] + stats["tiles_failed"]
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] processed={stats['tiles_processed']} "
                  f"skipped={stats['tiles_skipped']} failed={stats['tiles_failed']} "
                  f"({rate:.1f} tiles/s)", flush=True)

    stats["elapsed_seconds"] = time.time() - t0
    logger.info("Semantic preprocessing complete: %d processed, %d skipped, "
                "%d failed in %.1fs",
                stats["tiles_processed"], stats["tiles_skipped"],
                stats["tiles_failed"], stats["elapsed_seconds"])
    return stats


if __name__ == "__main__":
    import argparse
    from Dataset_Preprocessing import config as preproc_cfg

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Semantic reference tile preprocessor")
    parser.add_argument("--aerial-dir", type=Path,
                        default=preproc_cfg.REFERENCE_AERIAL_DIR)
    parser.add_argument("--prediction-dir", type=Path,
                        default=preproc_cfg.REFERENCE_PREDICTION_DIR)
    parser.add_argument("--model-path", type=Path,
                        default=preproc_cfg.SEMANTIC_MODEL_PATH)
    parser.add_argument("--device", default=preproc_cfg.DEVICE)
    parser.add_argument("--zoom", type=int, default=preproc_cfg.TMS_ZOOM_LEVEL)
    parser.add_argument("--force", action="store_true",
                        help="Re-process all tiles even if predictions exist")
    args = parser.parse_args()

    result = run_semantic_preprocessing(
        aerial_dir=args.aerial_dir,
        prediction_dir=args.prediction_dir,
        model_path=args.model_path,
        device=args.device,
        zoom=args.zoom,
        force=args.force,
    )
    print(f"\nDone: {result}")
