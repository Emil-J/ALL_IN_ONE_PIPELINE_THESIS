"""
SuperPoint Reference Tile Preprocessor.

Extracts and stores SuperPoint keypoints + descriptors for every reference
tile in the TMS aerial directory.  Results are stored in an HDF5 feature
store (see feature_store.py) for fast runtime loading.

Incremental: tiles already present in the store are skipped unless --force.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from Dataset_Preprocessing.feature_store import FeatureStoreWriter

logger = logging.getLogger(__name__)


def _discover_tiles(aerial_dir: Path, zoom: int = 16):
    """
    Walk the TMS aerial directory and yield (tile_x, tile_y, path).

    Expected structure: aerial_dir/{zoom}/{x}/{y}.png
    """
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


def _to_tensor_gray(image: np.ndarray, device: str) -> torch.Tensor:
    """Convert HWC uint8 image to (1, 1, H, W) float tensor for SuperPoint.

    Matches the conversion in Pipeline_3_Rev1/src/geometric_matcher.py
    exactly so that precomputed features are identical to runtime extraction.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    elif image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = image[..., 0].astype(np.float32)
    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def run_superpoint_preprocessing(
    aerial_dir: Path,
    output_h5_path: Path,
    *,
    max_keypoints: int = 2048,
    device: str = "cuda",
    zoom: int = 16,
    descriptor_dtype: str = "float32",
    force: bool = False,
) -> dict:
    """
    Extract SuperPoint features for all reference tiles and store in HDF5.

    Args:
        aerial_dir:      Path to TMS aerial tile directory.
        output_h5_path:  Path for the output HDF5 feature store.
        max_keypoints:   SuperPoint max keypoints (must match runtime config).
        device:          PyTorch device ("cuda" or "cpu").
        zoom:            TMS zoom level.
        descriptor_dtype: "float32" or "float16".
        force:           If True, re-extract even if tile already in store.

    Returns:
        dict with 'tiles_processed', 'tiles_skipped', 'tiles_failed',
        'total_tiles', 'elapsed_seconds'.
    """
    from lightglue import SuperPoint

    # Discover all tiles
    tiles = list(_discover_tiles(aerial_dir, zoom))
    total = len(tiles)
    logger.info("Discovered %d reference tiles in %s", total, aerial_dir)

    if total == 0:
        raise ValueError(f"No tiles found in {aerial_dir}/{zoom}/")

    # Initialize SuperPoint with same settings as runtime
    # Auto-detect device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    stats = {"tiles_processed": 0, "tiles_skipped": 0, "tiles_failed": 0,
             "total_tiles": total}

    t0 = time.time()

    with FeatureStoreWriter(
        output_h5_path,
        extractor_name="superpoint",
        max_keypoints=max_keypoints,
        tile_size_px=512,
        zoom=zoom,
        descriptor_dtype=descriptor_dtype,
    ) as writer:

        for i, (tx, ty, tile_path) in enumerate(tiles):
            # Skip if already in store (incremental)
            if not force and writer.has_tile(tx, ty):
                stats["tiles_skipped"] += 1
                continue

            try:
                # Load tile image
                img = np.array(Image.open(tile_path).convert("RGB"))
                h, w = img.shape[:2]

                # Extract features (same pipeline as runtime)
                with torch.no_grad():
                    tensor = _to_tensor_gray(img, device)
                    feats = extractor.extract(tensor)

                kpts = feats["keypoints"][0].cpu().numpy()       # (N, 2)
                descs = feats["descriptors"][0].cpu().numpy()     # (N, D)
                scores = feats["keypoint_scores"][0].cpu().numpy()  # (N,)

                writer.add_tile(tx, ty, kpts, descs, scores, h, w)
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
    logger.info("SuperPoint preprocessing complete: %d processed, %d skipped, "
                "%d failed in %.1fs",
                stats["tiles_processed"], stats["tiles_skipped"],
                stats["tiles_failed"], stats["elapsed_seconds"])
    return stats


if __name__ == "__main__":
    import argparse
    from Dataset_Preprocessing import config as preproc_cfg

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SuperPoint reference tile preprocessor")
    parser.add_argument("--aerial-dir", type=Path,
                        default=preproc_cfg.REFERENCE_AERIAL_DIR)
    parser.add_argument("--output", type=Path,
                        default=preproc_cfg.REFERENCE_FEATURES_PATH)
    parser.add_argument("--max-keypoints", type=int,
                        default=preproc_cfg.MAX_NUM_KEYPOINTS)
    parser.add_argument("--device", default=preproc_cfg.DEVICE)
    parser.add_argument("--zoom", type=int, default=preproc_cfg.TMS_ZOOM_LEVEL)
    parser.add_argument("--dtype", default=preproc_cfg.DESCRIPTOR_DTYPE,
                        choices=["float32", "float16"])
    parser.add_argument("--force", action="store_true",
                        help="Re-extract all tiles even if already in store")
    args = parser.parse_args()

    result = run_superpoint_preprocessing(
        aerial_dir=args.aerial_dir,
        output_h5_path=args.output,
        max_keypoints=args.max_keypoints,
        device=args.device,
        zoom=args.zoom,
        descriptor_dtype=args.dtype,
        force=args.force,
    )
    print(f"\nDone: {result}")
